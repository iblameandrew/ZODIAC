import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.optim import AdamW
import random
import numpy as np

# ==========================================
# 1. STATE OBJECTIVE ENGINE (MATH ANCHORS)
# ==========================================
class StateObjectives:
    """
    Translates various technical properties into differentiable PyTorch loss/reward functions.
    Vectorized to handle a batch of different objectives.
    """
    @staticmethod
    def get_reward(z_trace, history_trace, objective_labels, objective_indices):
        """
        z_trace: [Batch, Seq, Dim] - Current trajectory
        history_trace: [Batch, Window, Dim] - Past context
        objective_labels: List of available objective names
        objective_indices: [Batch] tensor of indices pointing to objective_labels
        """
        batch_size = z_trace.shape[0]
        rewards = torch.zeros(batch_size, device=z_trace.device)
        z_current = z_trace[:, -1, :] # Current state
        
        for i in range(batch_size):
            obj_name = objective_labels[objective_indices[i]]
            
            # 1. KINETIC_VELOCITY
            if obj_name == "KINETIC_VELOCITY":
                if z_trace.shape[1] < 2: rewards[i] = 0.0
                else:
                    velocity = torch.norm(z_current[i] - z_trace[i, -2, :], dim=-1)
                    rewards[i] = velocity * 2.0

            # 2. CENTROID_STABILITY
            elif obj_name == "CENTROID_STABILITY":
                if history_trace is None: rewards[i] = 0.0
                else:
                    centroid = history_trace[i].mean(dim=0)
                    dist = torch.norm(z_current[i] - centroid, dim=-1)
                    rewards[i] = -dist

            # 3. TEMPORAL_DUALITY
            elif obj_name == "TEMPORAL_DUALITY":
                if z_trace.shape[1] < 3: rewards[i] = 0.0
                else:
                    diff_1 = torch.norm(z_current[i] - z_trace[i, -2, :], dim=-1)
                    diff_2 = torch.norm(z_current[i] - z_trace[i, -3, :], dim=-1)
                    rewards[i] = diff_1 - (diff_2 * 0.5)

            # 4. CYCLIC_RECURRENCE
            elif obj_name == "CYCLIC_RECURRENCE":
                origin = z_trace[i, 0, :]
                sim = F.cosine_similarity(z_current[i].unsqueeze(0), origin.unsqueeze(0), dim=-1)
                rewards[i] = sim * 2.0

            # 5. REPRESENTATIVE_CENTRALITY
            elif obj_name == "REPRESENTATIVE_CENTRALITY":
                if batch_size > 1:
                    batch_mean = z_current.mean(dim=0)
                    sim = F.cosine_similarity(z_current[i].unsqueeze(0), batch_mean.unsqueeze(0), dim=-1)
                    mag = torch.norm(z_current[i], dim=-1)
                    rewards[i] = sim * mag
                else: rewards[i] = 0.0

            # 6. SPARSE_PRECISION
            elif obj_name == "SPARSE_PRECISION":
                l1_norm = torch.norm(z_current[i], p=1, dim=-1)
                rewards[i] = -l1_norm

            # 7. HARMONIC_EQUILIBRIUM
            elif obj_name == "HARMONIC_EQUILIBRIUM":
                if batch_size > 1:
                    center = z_current.mean(dim=0)
                    dist = torch.norm(z_current[i] - center, dim=-1)
                    rewards[i] = -dist
                else: rewards[i] = 0.0

            # 8. LATENT_ORTHOGONALITY
            elif obj_name == "LATENT_ORTHOGONALITY":
                if history_trace is None: rewards[i] = 0.0
                else:
                    surface = history_trace[i].mean(dim=0)
                    cosine = F.cosine_similarity(z_current[i].unsqueeze(0), surface.unsqueeze(0), dim=-1)
                    rewards[i] = 1.0 - torch.abs(cosine)

            # 9. VECTOR_EXPANSION
            elif obj_name == "VECTOR_EXPANSION":
                rewards[i] = torch.norm(z_current[i], dim=-1)

            # 10. STRUCTURAL_CONSTRAINT
            elif obj_name == "STRUCTURAL_CONSTRAINT":
                clamped = torch.clamp(z_current[i], -1.0, 1.0)
                dist_out = torch.norm(z_current[i] - clamped, dim=-1)
                rewards[i] = -dist_out * 10.0

            # 11. DIVERSITY_NOVELTY
            elif obj_name == "DIVERSITY_NOVELTY":
                if batch_size > 1:
                    center = z_current.mean(dim=0)
                    dist = torch.norm(z_current[i] - center, dim=-1)
                    rewards[i] = dist
                else: rewards[i] = 0.0

            # 12. ENTROPIC_DIFFUSION
            elif obj_name == "ENTROPIC_DIFFUSION":
                rewards[i] = -torch.max(torch.abs(z_current[i]), dim=-1).values

        # Numerical Stability: Handle NaNs in rewards
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        return rewards

# ==========================================
# 2. THE GRANULAR AGENT
# ==========================================
class GranularAgent(nn.Module):
    def __init__(self, dim, memory_size=512):
        super().__init__()
        self.dim = dim
        self.state_mu = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.verb_up = nn.Linear(dim, dim * 2)
        self.verb_act = nn.SiLU()
        self.verb_down = nn.Linear(dim * 2, dim)
        self.verb_gate = nn.Parameter(torch.zeros(1))
        self.memory_book = nn.Embedding(memory_size, dim)
        self.memory_book.weight.data.normal_(0, 0.02)
        
    def think(self, input_emb, neighbors=None):
        context = input_emb
        if neighbors is not None:
            context = context + (neighbors * 0.1)
        x = context + self.state_mu
        residual = x
        x = self.verb_up(x)
        x = self.verb_act(x)
        x = self.verb_down(x)
        x = residual + (x * torch.tanh(self.verb_gate))
        with torch.no_grad():
            dists = torch.cdist(x, self.memory_book.weight)
            indices = torch.argmin(dists, dim=-1)
        memory_vec = self.memory_book(indices)
        x = x + (memory_vec * 0.05)
        # Sanitization: Prevent NaNs from propagating
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Latent Norm Constraint
        tau = 10.0
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Avoid division by zero
        x = torch.where(norm > tau, x * (tau / torch.clamp(norm, min=1e-6)), x)
        return x

# ==========================================
# 3. THE WORLD MODEL (BACKBONE WRAPPER)
# ==========================================
class AgentWorldModel(nn.Module):
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        print(f">> Initializing Environment: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.backbone.parameters(): param.requires_grad = False
        self.dim = self.backbone.config.hidden_size
        self.root_agent = GranularAgent(self.dim).to(self.device).to(torch.float16)
        self.image_projector = nn.Sequential(
            nn.Linear(self.dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*64*4)
        ).to(self.device).to(torch.float16)

        self.objective_modes = [
            "KINETIC_VELOCITY", "CENTROID_STABILITY", "TEMPORAL_DUALITY", "CYCLIC_RECURRENCE", 
            "REPRESENTATIVE_CENTRALITY", "SPARSE_PRECISION", "HARMONIC_EQUILIBRIUM", "LATENT_ORTHOGONALITY", 
            "VECTOR_EXPANSION", "STRUCTURAL_CONSTRAINT", "DIVERSITY_NOVELTY", "ENTROPIC_DIFFUSION"
        ]

    def forward_agent(self, agent, token_emb, neighbor_context=None):
        return agent.think(token_emb, neighbor_context)

# ==========================================
# 4. SIMULATION LOOP (STABILIZED)
# ==========================================
def run_agent_simulation(world, keywords, episodes=50, max_len=40, K=4, recursion_depth=3):
    print(f">> Starting Agent Simulation with Keywords: {keywords}")
    
    params = list(world.root_agent.parameters()) + list(world.image_projector.parameters())
    optimizer = AdamW(params, lr=2e-4)
    
    # 1. Keyword Composition (Seed Embedding)
    with torch.no_grad():
        keyword_embs = []
        for word in keywords:
            ids = world.tokenizer(word, return_tensors="pt").input_ids.to(world.device)
            emb = world.backbone.model.embed_tokens(ids) # [1, Seq, Dim]
            keyword_embs.append(emb.mean(dim=1)) # Average tokens per word
        seed_emb = torch.stack(keyword_embs).mean(dim=0).unsqueeze(1) # [1, 1, Dim]
    
    # Initialize Agent Pool
    target_dtype = world.root_agent.verb_up.weight.dtype
    agent_pool = []
    
    for k in range(K):
        noise = torch.randn(1, world.dim, device=world.device, dtype=target_dtype) * 0.05
        state = world.root_agent.state_mu.clone() + noise
        agent_pool.append({
            "id": k,
            "parent_id": -1,
            "depth": 0,
            "state": state,
            "active": True,
            "tag": f"G0-A{k}"
        })
    
    active_indices = list(range(K))
    mating_threshold = 2.0 # Euclidean distance threshold for crossover
    
    for ep in range(episodes):
        optimizer.zero_grad()
        
        # Gather active states
        current_active_states = torch.cat([agent_pool[i]["state"] for i in active_indices], dim=0) # [K, Dim]
        curr_emb = seed_emb.repeat(len(active_indices), 1, 1) # [K, 1, Dim]
        
        trace_z = []
        log_probs_actions = []
        curr_ids = torch.empty(len(active_indices), 0, dtype=torch.long, device=world.device)
        
        # 2. Rotatory Objective Assignment
        obj_indices = torch.tensor([(k + ep) % 12 for k in range(len(active_indices))], device=world.device)
        
        # STABILIZATION HYPERPARAMETERS
        beta = 0.5 
        temperature = 0.6
        top_k_sampling = 40
        steering_strength = min(0.15, 0.05 + (ep * 0.01)) 
        EXPANDER_INDICES = {0, 2, 4, 8, 10, 11}
        kl_penalties = []
        
        for t in range(max_len):
            outputs = world.backbone.model(inputs_embeds=curr_emb, output_hidden_states=True)
            z_backbone = outputs.last_hidden_state[:, -1, :] 
            with torch.no_grad():
                base_logits = world.backbone.lm_head(z_backbone)
            
            z_backbone_half = z_backbone.to(target_dtype)
            
            # Gated Steering
            base_logits_32 = base_logits.float()
            probs_base = F.softmax(base_logits_32, dim=-1)
            log_probs_base = F.log_softmax(base_logits_32, dim=-1)
            entropy = -torch.sum(probs_base * log_probs_base, dim=-1)
            entropy = torch.nan_to_num(entropy, nan=0.0)
            steering_gate = torch.sigmoid((entropy - 2.5) * 2.0)
            
            # Forward Agent using its specific state from pool
            # We need to broadcast the forward_agent or loop?
            # Root agent uses its state_mu. We temporarily inject the pool state.
            
            thought_deltas = []
            for i, idx in enumerate(active_indices):
                # Temporary swap of root agent state_mu to specific agent state
                original_mu = world.root_agent.state_mu.data
                world.root_agent.state_mu.data = agent_pool[idx]["state"]
                
                delta = world.forward_agent(world.root_agent, z_backbone_half[[i]])
                thought_deltas.append(delta)
                
                world.root_agent.state_mu.data = original_mu
            
            thought_delta = torch.cat(thought_deltas, dim=0)
            thought_delta = torch.nan_to_num(thought_delta, nan=0.0)
            thought_delta = F.normalize(thought_delta, p=2, dim=-1)
            
            effective_strength = steering_strength * steering_gate.unsqueeze(-1)
            z_steered = z_backbone_half + (effective_strength * thought_delta)
            
            steered_logits = world.backbone.lm_head(z_steered.to(target_dtype))
            steered_logits = torch.nan_to_num(steered_logits, nan=0.0)
            
            blend_alpha = 0.3 * steering_gate.unsqueeze(-1)
            final_logits = ((1 - blend_alpha) * base_logits) + (blend_alpha * steered_logits)
            final_logits = torch.nan_to_num(final_logits, nan=0.0)

            trace_z.append(z_steered.unsqueeze(1))
            
            if t > 0:
                recent_ids = curr_ids[:, -5:] 
                final_logits.scatter_(1, recent_ids, -1.0)

            # Calculate KL for penalty (BEFORE Top-K mask)
            with torch.no_grad():
                 kl_step = F.kl_div(F.log_softmax(final_logits, dim=-1), probs_base, reduction='none', log_target=False).sum(dim=-1)
                 kl_step = torch.nan_to_num(kl_step, nan=10.0, posinf=10.0).clamp(max=20.0)
                 kl_penalties.append(kl_step)

            # Filter Logits (Top-K)
            if top_k_sampling > 0:
                v, _ = torch.topk(final_logits, top_k_sampling)
                out_mask = final_logits < v[:, [-1]]
                final_logits[out_mask] = float('-inf')

            probs = F.softmax(final_logits / temperature, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            if (probs.sum(dim=-1) == 0).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            next_token = torch.multinomial(probs, 1)
            token_log_prob = final_logits.gather(1, next_token)
            log_probs_actions.append(token_log_prob)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            with torch.no_grad():
                next_emb = world.backbone.model.embed_tokens(next_token)
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)
        
        full_trace = torch.cat(trace_z, dim=1)
        rewards = StateObjectives.get_reward(full_trace, None, world.objective_modes, obj_indices)
        mean_kl = torch.stack(kl_penalties, dim=1).mean(dim=1)
        rewards = rewards - (beta * mean_kl)
        rewards = torch.clamp(rewards, min=-50.0, max=50.0)
        
        # EVOLUTION: Recursive Lineage & Mating
        sorted_indices = torch.argsort(rewards, descending=True)
        num_survivors = len(active_indices) // 2
        top_local_indices = sorted_indices[:num_survivors]
        
        new_active_indices = []
        
        # 1. Crossover (Metric Mating)
        mated_pairs = set()
        for i in range(num_survivors):
            for j in range(i + 1, num_survivors):
                idx_i = active_indices[top_local_indices[i].item()]
                idx_j = active_indices[top_local_indices[j].item()]
                dist = torch.norm(agent_pool[idx_i]["state"] - agent_pool[idx_j]["state"])
                
                if dist < mating_threshold and len(new_active_indices) < K:
                    print(f">> Mating: Agents {agent_pool[idx_i]['tag']} and {agent_pool[idx_j]['tag']}...", flush=True)
                    # Crossover
                    child_state = (agent_pool[idx_i]["state"] + agent_pool[idx_j]["state"]) / 2.0
                    child_state += torch.randn_like(child_state) * 0.02 # Small mutation
                    
                    child_id = len(agent_pool)
                    agent_pool.append({
                        "id": child_id,
                        "parent_id": [idx_i, idx_j],
                        "depth": max(agent_pool[idx_i]["depth"], agent_pool[idx_j]["depth"]) + 1,
                        "state": child_state,
                        "active": True,
                        "tag": f"G{max(agent_pool[idx_i]['depth'], agent_pool[idx_j]['depth']) + 1}-C{idx_i}/{idx_j}"
                    })
                    new_active_indices.append(child_id)
                    mated_pairs.add(idx_i)
                    mated_pairs.add(idx_j)

        # 2. Sequential Reproduction for survivors
        for local_idx in top_local_indices:
            orig_idx = active_indices[local_idx.item()]
            if len(new_active_indices) < K:
                # Check recursion depth limit
                if agent_pool[orig_idx]["depth"] < recursion_depth:
                    # Parent freezes
                    agent_pool[orig_idx]["active"] = False
                    
                    # Spawn child
                    mutation = torch.randn_like(agent_pool[orig_idx]["state"]) * 0.05
                    child_state = agent_pool[orig_idx]["state"] + mutation
                    
                    child_id = len(agent_pool)
                    agent_pool.append({
                        "id": child_id,
                        "parent_id": orig_idx,
                        "depth": agent_pool[orig_idx]["depth"] + 1,
                        "state": child_state,
                        "active": True,
                        "tag": f"G{agent_pool[orig_idx]['depth'] + 1}-A{orig_idx}"
                    })
                    new_active_indices.append(child_id)
                else:
                    # Depth limit reached: Parent stays active instead of spawning
                    print(f">> Depth Limit Reached for {agent_pool[orig_idx]['tag']}. Staying active.", flush=True)
                    new_active_indices.append(orig_idx)
                    agent_pool[orig_idx]["active"] = True
            else:
                # If population is full, parent stays active
                agent_pool[orig_idx]["active"] = True
                new_active_indices.append(orig_idx)

        # Fill remaining slots if any
        while len(new_active_indices) < K:
            # Immigrant from root
            noise = torch.randn(1, world.dim, device=world.device, dtype=target_dtype) * 0.1
            state = world.root_agent.state_mu.clone() + noise
            child_id = len(agent_pool)
            agent_pool.append({
                "id": child_id, "parent_id": -1, "depth": 0, "state": state, "active": True, "tag": f"G0-N{child_id}"
            })
            new_active_indices.append(child_id)

        active_indices = new_active_indices
        
        # OPTIMIZATION (Policy Gradient)
        trajectory_log_probs = torch.cat(log_probs_actions, dim=1).sum(dim=1)
        if rewards.std() > 1e-6: adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        else: adv = rewards
        loss = -(adv * trajectory_log_probs).mean()
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
        
        # EPISODE END: MEMORY GATHERING (Consolidation)
        # Yield back descendants' intelligence to ancestors
        with torch.no_grad():
            for agent in reversed(agent_pool): # Bottom-up traversal
                if agent["parent_id"] != -1:
                    parents = agent["parent_id"] if isinstance(agent["parent_id"], list) else [agent["parent_id"]]
                    for p_idx in parents:
                        # Momentum update to parent state from child findings
                        alpha_yield = 0.05
                        agent_pool[p_idx]["state"].data = (1 - alpha_yield) * agent_pool[p_idx]["state"].data + alpha_yield * agent["state"].data
            
            # Final top-level gathering to root_agent state_mu
            # Only update root from Gen-0 agents or all?
            # User: "gathers all the childs and subchilds memory into the original parents memory".
            # The bottom-up loop handles this recursively. Gen-0 agents now hold the "yielded" intelligence.
            gen0_states = torch.cat([a["state"] for a in agent_pool if a["depth"] == 0], dim=0)
            centroid = gen0_states.mean(dim=0, keepdim=True).to(target_dtype)
            world.root_agent.state_mu.data = 0.9 * world.root_agent.state_mu.data + 0.1 * centroid

        best_idx = torch.argmax(rewards).item()
        best_agent_pool_idx = active_indices[best_idx]
        
        # LOGGING
        print(f"\nIteration {ep+1:02d} | Avg Reward: {rewards.mean().item():.4f} | Avg KL: {mean_kl.mean().item():.4f}", flush=True)
        for k, idx in enumerate(active_indices):
            agent_text = world.tokenizer.decode(curr_ids[k], skip_special_tokens=True).replace("\n", " ")
            mode_name = world.objective_modes[obj_indices[k]]
            safe_text = "".join(ch for ch in agent_text[:80] if ch.isprintable())
            is_best = "*" if k == best_idx else " "
            tag = agent_pool[idx]["tag"]
            print(f"   [{k}]{is_best} {tag:14} {mode_name[:20]:20} | R: {rewards[k]:6.2f} | \"{safe_text}...\"", flush=True)

    # Final Text generation from best agent
    best_text = world.tokenizer.decode(curr_ids[best_idx], skip_special_tokens=True)
    print("\n" + "="*50)
    print("FINAL COMPOSITION (Best Agent):")
    print(f"\"{best_text}\"")
    print("="*50)
    return best_text

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    KEYWORDS = ["ethereal", "cybernetic", "obsidian"]
    
    world_model = AgentWorldModel(MODEL_ID)
    
    run_agent_simulation(
        world_model, 
        keywords=KEYWORDS, 
        episodes=20, 
        max_len=40, 
        K=4,
        recursion_depth=3
    )
