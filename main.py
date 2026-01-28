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
def run_agent_simulation(world, keywords, episodes=50, max_len=40, K=4):
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
        
        # Mean of all keyword vectors
        seed_emb = torch.stack(keyword_embs).mean(dim=0).unsqueeze(1) # [1, 1, Dim]
    
    final_output = ""
    
    # Initialize Persistent Population
    target_dtype = world.root_agent.verb_up.weight.dtype
    noise = torch.randn(K, world.dim, device=world.device, dtype=target_dtype) * 0.05 # Reduced noise init
    population_states = world.root_agent.state_mu.expand(K, -1).clone() + noise
    
    agent_tags = ["GEN-0"] * K
    
    for ep in range(episodes):
        optimizer.zero_grad()
        
        curr_emb = seed_emb.repeat(K, 1, 1) # [K, 1, Dim]
        
        trace_z = []
        log_probs_actions = []
        curr_ids = torch.empty(K, 0, dtype=torch.long, device=world.device)
        
        # 2. Rotatory Objective Assignment
        obj_indices = torch.tensor([(k + ep) % 12 for k in range(K)], device=world.device)
        
        # STABILIZATION HYPERPARAMETERS
        beta = 0.5 
        temperature = 0.6  # Lower temp for coherence
        top_k_sampling = 40 # Cut off the long tail of junk tokens
        
        # Gentle steering ramp: Cap at 0.15 to prevent manifold collapse
        steering_strength = min(0.15, 0.05 + (ep * 0.01)) 
        
        EXPANDER_INDICES = {0, 2, 4, 8, 10, 11}
        kl_penalties = []
        
        for t in range(max_len):
            # -----------------------------------------------------------------
            # 1. DEEP BACKBONE PASS (Contextual Anchor)
            # -----------------------------------------------------------------
            outputs = world.backbone.model(inputs_embeds=curr_emb, output_hidden_states=True)
            z_backbone = outputs.last_hidden_state[:, -1, :] 
            
            with torch.no_grad():
                base_logits = world.backbone.lm_head(z_backbone)
            
            # -----------------------------------------------------------------
            # 2. RESIDUAL STEERING (Normalized & Blended)
            # -----------------------------------------------------------------
            target_dtype = world.root_agent.verb_up.weight.dtype
            z_backbone_half = z_backbone.to(target_dtype)
            population_states = population_states.to(target_dtype)
            
            # ENTROPY CALCULATION (Gate)
            base_logits_32 = base_logits.float()
            probs_base = F.softmax(base_logits_32, dim=-1)
            log_probs_base = F.log_softmax(base_logits_32, dim=-1)
            entropy = -torch.sum(probs_base * log_probs_base, dim=-1) # [K]
            entropy = torch.nan_to_num(entropy, nan=0.0)
            
            # Gate: Only steer when entropy is high (new word start)
            # Llama-3 average entropy is ~2.5. We gate around that.
            steering_gate = torch.sigmoid((entropy - 2.5) * 2.0)
            
            # Agent Thought
            thought_delta = world.forward_agent(world.root_agent, z_backbone_half, neighbor_context=population_states)
            
            # CRITICAL FIX 1: Normalize Delta 
            # Force agent output to be a direction unit vector.
            # Sanitize before normalization
            thought_delta = torch.nan_to_num(thought_delta, nan=0.0)
            thought_delta = F.normalize(thought_delta, p=2, dim=-1)
            
            # Apply Residual
            effective_strength = steering_strength * steering_gate.unsqueeze(-1)
            z_steered = z_backbone_half + (effective_strength * thought_delta)
            
            # -----------------------------------------------------------------
            # 3. LOGIT BLENDING (Grammar Preservation)
            # -----------------------------------------------------------------
            steered_logits = world.backbone.lm_head(z_steered.to(target_dtype))
            steered_logits = torch.nan_to_num(steered_logits, nan=0.0) # Harden
            
            # Blend factor
            blend_alpha = 0.3 * steering_gate.unsqueeze(-1)
            final_logits = ((1 - blend_alpha) * base_logits) + (blend_alpha * steered_logits)
            final_logits = torch.nan_to_num(final_logits, nan=0.0) # Harden against any NaN in base/steered

            trace_z.append(z_steered.unsqueeze(1))
            
            # -----------------------------------------------------------------
            # 4. SAMPLING WITH TOP-K FILTERING (The "Slicing" Fix)
            # -----------------------------------------------------------------
            
            # Repetition Penalty
            if t > 0:
                # Use scatter to penalize used tokens
                # Expand curr_ids to match logits dimension if needed, but usually [K, History]
                # We only penalize the immediate past to avoid locking up
                recent_ids = curr_ids[:, -5:] 
                final_logits.scatter_(1, recent_ids, -1.0) # Light penalty

            # Filter Logits (Top-K)
            # This removes the "long tail" of junk chars that cause mid-word slicing
            if top_k_sampling > 0:
                v, _ = torch.topk(final_logits, top_k_sampling)
                # Any logit smaller than the K-th largest gets -inf
                out_mask = final_logits < v[:, [-1]]
                final_logits[out_mask] = float('-inf')

            probs = F.softmax(final_logits / temperature, dim=-1)
            
            # Safety checks
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0)
            if (probs.sum(dim=-1) == 0).any():
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            idx_next = torch.multinomial(probs, 1)
            next_token = idx_next
            token_log_prob = final_logits.gather(1, next_token)
            log_probs_actions.append(token_log_prob)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            with torch.no_grad():
                next_emb = world.backbone.model.embed_tokens(next_token)
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)
            
            # Calculate KL for penalty
            with torch.no_grad():
                 kl_step = F.kl_div(F.log_softmax(final_logits, dim=-1), probs_base, reduction='none', log_target=False).sum(dim=-1)
                 kl_penalties.append(kl_step)
        
        full_trace = torch.cat(trace_z, dim=1)
        
        # 5. REWARD CALCULATION (Same as before)
        rewards = StateObjectives.get_reward(full_trace, None, world.objective_modes, obj_indices)
        mean_kl = torch.stack(kl_penalties, dim=1).mean(dim=1)
        mean_kl = torch.nan_to_num(mean_kl, nan=0.0) # Harden KL
        rewards = rewards - (beta * mean_kl)
        rewards = torch.nan_to_num(rewards, nan=0.0) # Harden Rewards
        rewards = torch.clamp(rewards, min=-50.0, max=50.0)
        
        # EVOLUTION (Same logic, compacted)
        if K > 1:
            sorted_indices = torch.argsort(rewards, descending=True)
            num_survivors = K // 2
            top_indices = sorted_indices[:num_survivors]
            print(f">> Evolution: Agents {top_indices.tolist()} spawning children...", flush=True)
            survivor_states = full_trace[top_indices, -1, :].detach()
            mutation = torch.randn_like(survivor_states) * 0.05 # Reduced mutation
            offspring = survivor_states + mutation
            new_pop = torch.cat([survivor_states, offspring], dim=0)
            if new_pop.shape[0] < K:
                 new_pop = torch.cat([new_pop, survivor_states[:K-new_pop.shape[0]]], dim=0)
            population_states = new_pop[:K]
            
            survivor_tags = [agent_tags[i] for i in top_indices.tolist()]
            agent_tags = survivor_tags + [f"CHILD-Ep{ep+1}"] * num_survivors
            if len(agent_tags) < K: agent_tags += survivor_tags[:(K - len(agent_tags))]

        # OPTIMIZATION
        trajectory_log_probs = torch.cat(log_probs_actions, dim=1).sum(dim=1)
        if K > 1 and rewards.std() > 0: adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        else: adv = rewards
        loss = -(adv * trajectory_log_probs).mean()
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
        
        # BACKPASS
        with torch.no_grad():
            final_states = full_trace[:, -1, :]
            weights = F.softmax(rewards, dim=0)
            centroid = (weights.unsqueeze(1) * final_states).sum(dim=0).to(target_dtype)
            world.root_agent.state_mu.data = 0.9 * world.root_agent.state_mu.data + 0.1 * centroid.unsqueeze(0)
        
        best_idx = torch.argmax(rewards).item()
        
        # LOGGING
        print(f"\nIteration {ep+1:02d} | Avg Reward: {rewards.mean().item():.4f} | Avg KL: {mean_kl.mean().item():.4f}", flush=True)
        for k in range(K):
            agent_text = world.tokenizer.decode(curr_ids[k], skip_special_tokens=True).replace("\n", " ").replace("\r", " ")
            mode_name = world.objective_modes[obj_indices[k]]
            safe_text = "".join(ch for ch in agent_text[:80] if ch.isprintable())
            is_best = "*" if k == best_idx else " "
            print(f"   [{k}]{is_best} {agent_tags[k]:12} {mode_name[:20]:20} | R: {rewards[k]:6.2f} | \"{safe_text}...\"", flush=True)

    # Final Text generation from best agent in last episode
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
        max_len=30, 
        K=4 
    )
