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
# 4. SIMULATION LOOP
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
    # [K, Dim]
    # Ensure dtype matches model (Float16)
    target_dtype = world.root_agent.verb_up.weight.dtype
    noise = torch.randn(K, world.dim, device=world.device, dtype=target_dtype) * 0.1
    noise = torch.randn(K, world.dim, device=world.device, dtype=target_dtype) * 0.1
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
        
        # Hyperparameters
        beta = 0.5 
        temperature = 0.7
        steering_strength = min(0.15, ep * 0.02)
        top_k = 50 # Required for Fusion loop
        # EXPANDERS: Aries(0), Gemini(2), Leo(4), Sagit(8), Aquar(10), Pisces(11)
        EXPANDER_INDICES = {0, 2, 4, 8, 10, 11}
        
        kl_penalties = []
        # swarm_interactions removed
        
        for t in range(max_len):
            # -----------------------------------------------------------------
            # 1. DEEP BACKBONE PASS (Corrected Inference)
            # -----------------------------------------------------------------
            outputs = world.backbone.model(inputs_embeds=curr_emb, output_hidden_states=True)
            z_backbone = outputs.last_hidden_state[:, -1, :] 
            
            with torch.no_grad():
                base_logits = world.backbone.lm_head(z_backbone)
            
            # -----------------------------------------------------------------
            # 2. RESIDUAL STEERING (Agent as Delta)
            # -----------------------------------------------------------------
            target_dtype = world.root_agent.verb_up.weight.dtype
            z_backbone_half = z_backbone.to(target_dtype)
            population_states = population_states.to(target_dtype)
            
            # ENTROPY-GATED STEERING
            # Prevent "mid-word slicing" by only steering when model is uncertain
            # Cast to float32 to prevent numerical instability (NaNs from 0*-inf)
            base_logits_32 = base_logits.float()
            probs_base = F.softmax(base_logits_32, dim=-1)
            log_probs_base = F.log_softmax(base_logits_32, dim=-1)
            entropy = -torch.sum(probs_base * log_probs_base, dim=-1) # [K]
            entropy = torch.nan_to_num(entropy, nan=0.0)
            
            # Gate: High entropy (~5.0) = New Word -> Gate=1.0. Low entropy -> Gate=0.0
            steering_gate = torch.sigmoid(entropy - 3.0)
            
            thought_delta = world.forward_agent(world.root_agent, z_backbone_half, neighbor_context=population_states)
            
            # z_steered = z_backbone + (strength * gate * delta)
            effective_strength = steering_strength * steering_gate.unsqueeze(-1)
            z_steered = z_backbone_half + (effective_strength * thought_delta)
            
            # 3. GEOMETRIC ERROR CORRECTION
            if K > 1:
                center = z_steered.mean(dim=0, keepdim=True)
                dist_vec = z_steered - center
                expander_mask = torch.tensor([idx.item() in EXPANDER_INDICES for idx in obj_indices], device=world.device).unsqueeze(1)
                reducer_mask = ~expander_mask
                
                correction = torch.zeros_like(z_steered)
                correction[expander_mask.squeeze(1)] = dist_vec[expander_mask.squeeze(1)] * 0.1
                correction[reducer_mask.squeeze(1)] = -dist_vec[reducer_mask.squeeze(1)] * 0.1
                z_steered = z_steered + correction
                
                norm = torch.norm(z_steered, dim=-1, keepdim=True)
                z_steered = torch.where(norm > 15.0, z_steered * (15.0 / torch.clamp(norm, min=1e-6)), z_steered)
            
            trace_z.append(z_steered.unsqueeze(1))
            final_logits = world.backbone.lm_head(z_steered.to(target_dtype))
            
            # COHERENCE GUARDRAIL (KL Cutoff)
            # If agent goes too far off-manifold, revert proportional to violation
            with torch.no_grad():
                 kl_check = F.kl_div(F.log_softmax(final_logits, dim=-1), probs_base, reduction='none').sum(dim=-1) # [K]
                 mask_kl = kl_check > 1.5
                 if mask_kl.any():
                      # Dampen the steered vector towards backbone
                      z_steered[mask_kl] = (z_steered[mask_kl] * 0.5) + (z_backbone_half[mask_kl] * 0.5)
                      # Recompute logits
                      final_logits = world.backbone.lm_head(z_steered.to(target_dtype))
            
            with torch.no_grad():
                 base_probs = F.softmax(base_logits / temperature, dim=-1)
                 kl_step = F.kl_div(final_logits, base_probs, reduction='none', log_target=False).sum(dim=-1)
                 kl_penalties.append(kl_step)
            
            # Repetition Penalty
            if t > 0:
                penalty_value = 2.0
                fake_scores = torch.ones_like(curr_ids, dtype=final_logits.dtype) * (-penalty_value)
                final_logits.scatter_add_(1, curr_ids, fake_scores)

            probs = F.softmax(final_logits / temperature, dim=-1)
            if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                probs = torch.nan_to_num(probs, nan=0.0)
                if probs.sum() == 0: probs = torch.ones_like(probs) / probs.shape[-1]
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            idx_next = torch.multinomial(probs, 1)
            next_token = idx_next
            token_log_prob = final_logits.gather(1, next_token)
            log_probs_actions.append(token_log_prob)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            with torch.no_grad():
                next_emb = world.backbone.model.embed_tokens(next_token)
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)
        
        full_trace = torch.cat(trace_z, dim=1)
        
        # 3. Vectorized Reward Evaluation
        rewards = StateObjectives.get_reward(full_trace, None, world.objective_modes, obj_indices)
        
        # Apply KL Penalty
        mean_kl = torch.stack(kl_penalties, dim=1).mean(dim=1) # [K]
        rewards = rewards - (beta * mean_kl)
        
        # Reward Clamping: Prevent gradient explosions from outlier rewards
        rewards = torch.clamp(rewards, min=-50.0, max=50.0)
        
        # EVOLUTIONARY STEP (Reproduction)
        # Sort agents by reward
        if K > 1:
            sorted_indices = torch.argsort(rewards, descending=True)
            num_survivors = K // 2
            top_indices = sorted_indices[:num_survivors]
            
            # Log Evolution
            print(f">> Evolution: Agents {top_indices.tolist()} spawning children...", flush=True)
            
            # Select Survivors (Final state of thought vector)
            survivor_states = full_trace[top_indices, -1, :].detach() # [Survivors, Dim]
            
            # Create Offspring: Mutate survivors
            mutation_noise = torch.randn_like(survivor_states) * 0.1
            offspring_states = survivor_states + mutation_noise
            
            # Refill population
            new_pop = torch.cat([survivor_states, offspring_states], dim=0)
            
            # If K is odd, handle truncation/padding
            if new_pop.shape[0] < K:
                 needed = K - new_pop.shape[0]
                 extra = survivor_states[:needed]
                 new_pop = torch.cat([new_pop, extra], dim=0)
                 # Should extend tags too, but simplifying for even K
            
            population_states = new_pop[:K] # Update persistent population for next ep
            
            # Update Lineage Tags
            survivor_tags = [agent_tags[i] for i in top_indices.tolist()]
            child_tags = [f"CHILD-Ep{ep+1}"] * num_survivors
            agent_tags = survivor_tags + child_tags
            # Handle Odd K padding in tags if needed
            if len(agent_tags) < K:
                agent_tags += survivor_tags[:(K - len(agent_tags))]
        
        trajectory_log_probs = torch.cat(log_probs_actions, dim=1).sum(dim=1)
        
        if K > 1 and rewards.std() > 0:
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        else:
            adv = rewards
            
        loss = -(adv * trajectory_log_probs).mean()
        
        # NaN Guard: Do not backpropagate NaNs
        if torch.isnan(loss):
            print(f"      [WARNING] NaN loss detected, skipping backprop", flush=True)
            optimizer.zero_grad()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5) # Stricter clipping
            optimizer.step()
        
        # 4. Knowledge Integration Backpass (Back-Pollination)
        with torch.no_grad():
            final_states = full_trace[:, -1, :] # [K, Dim]
            # Weighted average based on rewards
            weights = F.softmax(rewards, dim=0)
            centroid = (weights.unsqueeze(1) * final_states).sum(dim=0)
            
            # Soft update root agent
            alpha_backpass = 0.1
            
            # CRITICAL: Ensure we stick to float16 to prevent dtype mismatch in next iteration
            target_dtype = world.root_agent.verb_up.weight.dtype 
            centroid = centroid.to(target_dtype)
            
            world.root_agent.state_mu.data = (1 - alpha_backpass) * world.root_agent.state_mu.data + alpha_backpass * centroid.unsqueeze(0)
        
        best_idx = torch.argmax(rewards).item()
        
        if ep == episodes - 1:
            # 5. Compositional Fusion
            print(">> Generating Compositional Fusion...")
            avg_reward = rewards.mean()
            successful_mask = rewards >= avg_reward
            
            if successful_mask.sum() > 0:
                # fused_state logic removed as it was unused and root_agent is already updated via backpass
                fused_ids = []
                fused_emb = seed_emb[[0]].clone() # [1, 1, Dim] from first seed
                
                tau = 10.0
                curr_fusion_emb = fused_emb
                
                for _ in range(max_len):
                    curr_input = curr_fusion_emb[:, -1, :]
                    curr_input = curr_input.to(world.root_agent.verb_up.weight.dtype) # Cast for safety
                    
                    # Get base logits for grammar
                    with torch.no_grad():
                        base_logits = world.backbone.lm_head(curr_input)
                    
                    # Get agent logits for intent
                    thought = world.forward_agent(world.root_agent, curr_input)
                    # Redundant norm constraint removed (already in think)
                    agent_logits = world.backbone.lm_head(thought)
                    
                    # LOGIT FUSION for Compositional Fusion
                    logits = base_logits + (steering_strength * agent_logits)
                    logits = torch.clamp(logits, min=-100.0, max=100.0)
                    
                    # REPETITION PENALTY for Fusion
                    if len(fused_ids) > 0:
                        for prev_token_tensor in fused_ids:
                             logits[0, prev_token_tensor.item()] -= 2.0
                    
                    # TOP-K FILTERING
                    final_logits_filtered = logits.clone()
                    top_k_values, _ = torch.topk(final_logits_filtered, top_k, dim=-1)
                    threshold = top_k_values[:, -1].unsqueeze(-1)
                    final_logits_filtered[final_logits_filtered < threshold] = float('-inf')
                    
                    # Sampling with Temperature
                    probs = F.softmax(final_logits_filtered / temperature, dim=-1)
                    
                    # Robust Probability Handling (same as main loop)
                    if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                        probs = torch.nan_to_num(probs, nan=0.0)
                        if probs.sum() == 0:
                            probs = torch.ones_like(probs) / probs.shape[-1]
                    
                    probs = probs + 1e-10
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    next_token = torch.multinomial(probs, 1)
                    fused_ids.append(next_token)
                    
                    with torch.no_grad():
                         next_emb_vec = world.backbone.model.embed_tokens(next_token)
                    curr_fusion_emb = torch.cat([curr_fusion_emb, next_emb_vec], dim=1)
                
                final_output = world.tokenizer.decode(torch.cat(fused_ids, dim=1).squeeze(0), skip_special_tokens=True)
            else:
                final_output = world.tokenizer.decode(curr_ids[best_idx], skip_special_tokens=True)
            
            print(f"\nIteration {ep+1:02d} | Avg Reward: {rewards.mean().item():.4f} | Avg KL: {mean_kl.mean().item():.4f}", flush=True)
            for k in range(K):
                agent_text = world.tokenizer.decode(curr_ids[k], skip_special_tokens=True).replace("\n", " ").replace("\r", " ")
                mode_name = world.objective_modes[obj_indices[k]]
                reward_val = rewards[k].item()
                is_best = "*" if k == best_idx else " "
                # Sanitize text for console printing regarding encoding and control chars
                safe_text = "".join(ch for ch in agent_text[:60] if ch.isprintable())
                polarity = "[EXPAND]" if obj_indices[k].item() in EXPANDER_INDICES else "[REDUCE]"
                tag = agent_tags[k]
                print(f"   [{k}]{is_best} {tag:12} Mode: {mode_name:25} {polarity} | Reward: {reward_val:8.4f} | \"{safe_text}...\"", flush=True)
        
        # Print iteration for ALL episodes (moved outside the if block)
        else:
            print(f"\nIteration {ep+1:02d} | Avg Reward: {rewards.mean().item():.4f} | Avg KL: {mean_kl.mean().item():.4f}", flush=True)
            for k in range(K):
                agent_text = world.tokenizer.decode(curr_ids[k], skip_special_tokens=True).replace("\n", " ").replace("\r", " ")
                mode_name = world.objective_modes[obj_indices[k]]
                reward_val = rewards[k].item()
                is_best = "*" if k == best_idx else " "
                safe_text = "".join(ch for ch in agent_text[:60] if ch.isprintable())
                polarity = "[EXPAND]" if obj_indices[k].item() in EXPANDER_INDICES else "[REDUCE]"
                tag = agent_tags[k]
                print(f"   [{k}]{is_best} {tag:12} Mode: {mode_name:25} {polarity} | Reward: {reward_val:8.4f} | \"{safe_text}...\"", flush=True)

    print("\n" + "="*50)
    print("FINAL COMPOSITION (Best Agent):")
    print(f"\"{final_output}\"")
    print("="*50)
    return final_output

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
