from dataclasses import dataclass, field
from typing import List, Optional
import asyncio
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel

# ==========================================
# 0. CONFIG & CONSTANTS
# ==========================================

class SimParams(BaseModel):
    keywords: List[str] = ["ethereal", "cybernetic", "obsidian"]
    max_steps: int = 50
    K: int = 4
    test_mode: bool = False

OBJECTIVE_COLORS = {
    "KINETIC_VELOCITY": "#00FFFF",  # Cyan
    "CENTROID_STABILITY": "#FF00FF",  # Magenta
    "TEMPORAL_DUALITY": "#FFFF00",  # Yellow
    "CYCLIC_RECURRENCE": "#00FF00",  # Lime
    "REPRESENTATIVE_CENTRALITY": "#FF6B6B",  # Coral Red
    "SPARSE_PRECISION": "#9B59B6",  # Purple
    "HARMONIC_EQUILIBRIUM": "#3498DB",  # Blue
    "LATENT_ORTHOGONALITY": "#E74C3C",  # Red
    "VECTOR_EXPANSION": "#F39C12",  # Orange
    "STRUCTURAL_CONSTRAINT": "#1ABC9C",  # Teal
    "DIVERSITY_NOVELTY": "#E91E63",  # Pink
    "ENTROPIC_DIFFUSION": "#8E44AD",  # Deep Purple
}

# ==========================================
# 1. CORE MODELS
# ==========================================

class StateObjectives:
    """
    Translates various technical properties into differentiable PyTorch loss/reward functions.
    Vectorized to handle a batch of different objectives.
    """
    @staticmethod
    def get_reward(z_trace, history_trace, objective_labels, objective_indices):
        batch_size = z_trace.shape[0]
        rewards = torch.zeros(batch_size, device=z_trace.device)
        z_current = z_trace[:, -1, :]  # Current state

        for i in range(batch_size):
            obj_name = objective_labels[objective_indices[i]]

            # 1. KINETIC_VELOCITY
            if obj_name == "KINETIC_VELOCITY":
                if z_trace.shape[1] < 2:
                    rewards[i] = 0.0
                else:
                    velocity = torch.norm(z_current[i] - z_trace[i, -2, :], dim=-1)
                    rewards[i] = velocity * 2.0

            # 2. CENTROID_STABILITY
            elif obj_name == "CENTROID_STABILITY":
                if history_trace is None:
                    rewards[i] = 0.0
                else:
                    centroid = history_trace[i].mean(dim=0)
                    dist = torch.norm(z_current[i] - centroid, dim=-1)
                    rewards[i] = -dist

            # 3. TEMPORAL_DUALITY
            elif obj_name == "TEMPORAL_DUALITY":
                if z_trace.shape[1] < 3:
                    rewards[i] = 0.0
                else:
                    diff_1 = torch.norm(z_current[i] - z_trace[i, -2, :], dim=-1)
                    diff_2 = torch.norm(z_current[i] - z_trace[i, -3, :], dim=-1)
                    rewards[i] = diff_1 - (diff_2 * 0.5)

            # 4. CYCLIC_RECURRENCE
            elif obj_name == "CYCLIC_RECURRENCE":
                origin = z_trace[i, 0, :]
                sim = F.cosine_similarity(
                    z_current[i].unsqueeze(0), origin.unsqueeze(0), dim=-1
                )
                rewards[i] = sim * 2.0

            # 5. REPRESENTATIVE_CENTRALITY
            elif obj_name == "REPRESENTATIVE_CENTRALITY":
                if batch_size > 1:
                    batch_mean = z_current.mean(dim=0)
                    sim = F.cosine_similarity(
                        z_current[i].unsqueeze(0), batch_mean.unsqueeze(0), dim=-1
                    )
                    mag = torch.norm(z_current[i], dim=-1)
                    rewards[i] = sim * mag
                else:
                    rewards[i] = 0.0

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
                else:
                    rewards[i] = 0.0

            # 8. LATENT_ORTHOGONALITY
            elif obj_name == "LATENT_ORTHOGONALITY":
                if history_trace is None:
                    rewards[i] = 0.0
                else:
                    surface = history_trace[i].mean(dim=0)
                    cosine = F.cosine_similarity(
                        z_current[i].unsqueeze(0), surface.unsqueeze(0), dim=-1
                    )
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
                else:
                    rewards[i] = 0.0

            # 12. ENTROPIC_DIFFUSION
            elif obj_name == "ENTROPIC_DIFFUSION":
                rewards[i] = -torch.max(torch.abs(z_current[i]), dim=-1).values

        # Numerical Stability
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)
        return rewards


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
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Latent Norm Constraint
        tau = 10.0
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = torch.where(norm > tau, x * (tau / torch.clamp(norm, min=1e-6)), x)
        return x


class AgentWorldModel(nn.Module):
    def __init__(self, model_id="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        print(f">> Initializing Environment: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.dim = self.backbone.config.hidden_size
        self.root_agent = GranularAgent(self.dim).to(self.device).to(torch.float16)
        
        self.objective_modes = [
            "KINETIC_VELOCITY", "CENTROID_STABILITY", "TEMPORAL_DUALITY",
            "CYCLIC_RECURRENCE", "REPRESENTATIVE_CENTRALITY", "SPARSE_PRECISION",
            "HARMONIC_EQUILIBRIUM", "LATENT_ORTHOGONALITY", "VECTOR_EXPANSION",
            "STRUCTURAL_CONSTRAINT", "DIVERSITY_NOVELTY", "ENTROPIC_DIFFUSION",
        ]

    def forward_agent(self, agent, token_emb, neighbor_context=None):
        return agent.think(token_emb, neighbor_context)


# ==========================================
# 2. SIMULATION STATE & RUNNER
# ==========================================

@dataclass
class SimState:
    running: bool = False
    current_step: int = 0
    max_steps: int = 40
    frame_buffer: list = field(default_factory=list)
    pca: Optional[PCA] = None
    min_vals: Optional[np.ndarray] = None
    max_vals: Optional[np.ndarray] = None
    
    # EMA for variance scaling
    ema_std: float = 1.0
    ema_alpha: float = 0.1  # Smoothing factor
    
    # Persistent logic state
    world_model: Optional[AgentWorldModel] = None
    optimizer: Optional[AdamW] = None
    agent_pool: list = field(default_factory=list)
    episode_count: int = 0


# Global State Instance
state = SimState()

async def zodiac_simulation(params: SimParams):
    """
    Main simulation generator. Yields frames for frontend or CLI.
    """
    global state

    state.running = True
    state.current_step = 0
    state.frame_buffer = []
    state.pca = None  # Force reset of PCA
    
    # Initialize World Model if needed
    if state.world_model is None:
        state.world_model = AgentWorldModel()
        state.optimizer = AdamW(state.world_model.root_agent.parameters(), lr=2e-4)
    
    world = state.world_model
    device = world.device
    target_dtype = torch.float16

    # 1. Keyword Composition (Seed Embedding)
    with torch.no_grad():
        keyword_embs = []
        for word in params.keywords:
            ids = world.tokenizer(word, return_tensors="pt").input_ids.to(device)
            emb = world.backbone.model.embed_tokens(ids)
            keyword_embs.append(emb.mean(dim=1))
        seed_emb = torch.stack(keyword_embs).mean(dim=0).unsqueeze(1)  # [1, 1, Dim]

    # Initialize Agent Pool
    if not state.agent_pool or len(state.agent_pool) != params.K:
        print(f">> Initializing new agent pool at the Heart: K={params.K}")
        state.agent_pool = []
        for k in range(params.K):
            # Noise reduced from 0.05 to 0.01 to start at the "Heart"
            noise = torch.randn(1, world.dim, device=device, dtype=target_dtype) * 0.01
            agent_state = world.root_agent.state_mu.clone() + noise
            state.agent_pool.append({
                "id": k,
                "state": agent_state,
                "tag": f"A{k}"
            })
    
    active_indices = list(range(params.K))
    curr_emb = seed_emb.repeat(params.K, 1, 1)
    
    # Trackers for this episode
    curr_ids = torch.empty(params.K, 0, dtype=torch.long, device=device)
    trace_z = []
    log_probs_actions = []
    kl_penalties = []
    
    # Assign initial objectives
    obj_indices = torch.tensor(
        [(k + state.episode_count) % len(world.objective_modes) for k in range(params.K)], 
        device=device
    )

    THOUGHT_HORIZON = 8  # Tokens per thought frame

    for t in range(params.max_steps):
        if not state.running:
            break
            
        state.current_step = t
        
        # Accumulators for this thought step
        thought_texts = [""] * params.K
        avg_thought_states = torch.zeros(params.K, world.dim, device=device, dtype=target_dtype)
        
        # --- INNER LOOP: GENERATE THOUGHT CHUNK ---
        for micro_t in range(THOUGHT_HORIZON):
            world.backbone.model.eval() 
            
            # 1. Backbone Forward
            outputs = world.backbone.model(inputs_embeds=curr_emb, output_hidden_states=True)
            z_backbone = outputs.last_hidden_state[:, -1, :]
            
            with torch.no_grad():
                base_logits = world.backbone.lm_head(z_backbone)
                
            z_backbone_half = z_backbone.to(target_dtype)
            
            # 2. Gated Steering
            base_logits_32 = base_logits.float()
            probs_base = F.softmax(base_logits_32, dim=-1)
            log_probs_base = F.log_softmax(base_logits_32, dim=-1)
            entropy = -torch.sum(probs_base * log_probs_base, dim=-1)
            entropy = torch.nan_to_num(entropy, nan=0.0)
            steering_gate = torch.sigmoid((entropy - 2.5) * 2.0)
            
            # 3. Agent Thinking (Steering)
            state.optimizer.zero_grad() 
            thought_deltas = []
            plasticity = 0.1
            
            for i, idx in enumerate(active_indices):
                current_agent_state = state.agent_pool[idx]["state"]
                original_magnitude = current_agent_state.norm(dim=-1, keepdim=True)
                
                # Swap root agent state
                original_mu = world.root_agent.state_mu.data
                world.root_agent.state_mu.data = current_agent_state
                
                # Think
                delta = world.forward_agent(world.root_agent, z_backbone_half[[i]])
                thought_deltas.append(delta)
                
                # Evolve state (detached)
                new_state = current_agent_state + (delta.detach() * plasticity)
                new_magnitude = new_state.norm(dim=-1, keepdim=True)
                new_state = new_state * (original_magnitude / (new_magnitude + 1e-8))
                state.agent_pool[idx]["state"] = new_state
                
                # Restore root
                world.root_agent.state_mu.data = original_mu

            thought_delta = torch.cat(thought_deltas, dim=0)
            thought_delta = torch.nan_to_num(thought_delta, nan=0.0)
            thought_delta = F.normalize(thought_delta, p=2, dim=-1)
            
            steering_strength = 0.15
            effective_strength = steering_strength * steering_gate.unsqueeze(-1)
            z_steered = z_backbone_half + (effective_strength * thought_delta)
            
            # Accumulate state for Viz
            avg_thought_states += z_steered
            
            # 4. Final Logits
            steered_logits = world.backbone.lm_head(z_steered.to(target_dtype))
            blend_alpha = 0.3 * steering_gate.unsqueeze(-1)
            final_logits = ((1 - blend_alpha) * base_logits) + (blend_alpha * steered_logits)
            final_logits = torch.nan_to_num(final_logits, nan=0.0)
            
            trace_z.append(z_steered.unsqueeze(1))
            
            # 5. KL Calc
            with torch.no_grad():
                kl_step = F.kl_div(
                    F.log_softmax(final_logits, dim=-1),
                    probs_base,
                    reduction="none",
                    log_target=False
                ).sum(dim=-1)
                kl_step = torch.nan_to_num(kl_step, nan=10.0, posinf=10.0).clamp(max=20.0)
                kl_penalties.append(kl_step)

            # 6. Sampling with Repetition Penalty & Special Token Suppression
            # Filter special tokens (Comprehensive search for <|...|>)
            if not hasattr(state, "bad_words_ids"):
                bad_ids = set()
                # Suppress known Llama 3 special tokens and patterns
                for i in range(min(world.tokenizer.vocab_size, 128500)):
                    try:
                        t_str = world.tokenizer.convert_ids_to_tokens(i)
                        if t_str and ("<|" in t_str or "|>" in t_str or "==" in t_str):
                            bad_ids.add(i)
                    except:
                        continue
                state.bad_words_ids = list(bad_ids)

            final_logits[:, state.bad_words_ids] = float("-inf")
            
            # Repetition Penalty (Stronger and broader)
            if curr_ids.shape[1] > 0:
                 for k in range(params.K):
                     # Unique tokens in the whole current context
                     recent_context = curr_ids[k, -100:] 
                     unique_ids, counts = torch.unique(recent_context, return_counts=True)
                     # Apply penalty proportional to frequency
                     final_logits[k, unique_ids] -= (counts.float() * 3.0) 

            top_k = 30
            v, _ = torch.topk(final_logits, top_k)
            final_logits[final_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(final_logits / 0.8, dim=-1) 
            probs = torch.nan_to_num(probs, nan=0.0)
            
            next_token = torch.multinomial(probs, 1)
            token_log_prob = final_logits.gather(1, next_token)
            log_probs_actions.append(token_log_prob)
            
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            with torch.no_grad():
                next_emb = world.backbone.model.embed_tokens(next_token)
            curr_emb = torch.cat([curr_emb, next_emb], dim=1)
            
            # Decode Token
            for k in range(params.K):
                token_str = world.tokenizer.decode(next_token[k])
                thought_texts[k] += token_str
                
        # --- END INNER LOOP ---
        
        # Calculate Per-Step Rewards to identifies "Best Agent"
        # We use the current chunk's latent trace to see who is leading.
        chunk_trace = z_steered.unsqueeze(1) # [K, 1, Dim] for the logic in get_reward
        # Note: get_reward expects a trace. For a single-step "best", we just look at this chunk.
        step_rewards = StateObjectives.get_reward(chunk_trace, None, world.objective_modes, obj_indices)
        best_agent_idx = torch.argmax(step_rewards).item()

        # Average state for this chunk for visualization
        avg_thought_states = avg_thought_states / THOUGHT_HORIZON
        avg_states_np = avg_thought_states.detach().float().cpu().numpy()

        # Cycle Objectives
        obj_indices = [(idx + 1) % len(world.objective_modes) for idx in obj_indices]

        # Centering for Robust PCA
        # Agents inevitably drift in the high-dim space.
        # We subtract the mean of the current batch to visualize only RELATIVE positions (shape).
        center_vec = np.mean(avg_states_np, axis=0, keepdims=True)
        centered_states = avg_states_np #- center_vec 
        # Actually, let's keep it simple: Fit PCA on t=0 centered data, then project centered data.
        # But wait, standard PCA subtracts the training mean.
        # If we subtract the CURRENT mean, we effectively ignore translation.
        # Yes, that is what we want.

        if state.pca is None or t == 0:
            state.pca = PCA(n_components=3)
            # Re-seed PCA with dummy noise to define a basis around the origin
            dummy_data = np.random.randn(10, world.dim) * 0.1
            # Fit on dummy data + centered initial state (which is roughly 0 mean).
            # No, let's fit on the *actual* centered initial state so the axes retain meaning.
            centered_init = avg_states_np - np.mean(avg_states_np, axis=0, keepdims=True)
            state.pca.fit(np.vstack([centered_init, dummy_data])) 
        
        # Projects CENTEREED and NORMALIZED states
        # Subtract mean (Translation invariant)
        center_vec = np.mean(avg_states_np, axis=0, keepdims=True)
        centered_current = avg_states_np - center_vec
        
        # Scale by variance (Scale invariant with EMA for smoothness)
        # This ensures that even if agents move very little, they expand to fill the view.
        # Conversely, if they explode apart, they are squeezed back into the view.
        curr_std = np.std(centered_current)
        
        # Initialize or Update EMA
        if t == 0:
            state.ema_std = max(curr_std, 1.5) # Scale Floor increased to 1.5 (Gravity)
        else:
            state.ema_std = (state.ema_alpha * curr_std) + ((1 - state.ema_alpha) * state.ema_std)
        
        # Apply scaling with floor
        effective_std = max(state.ema_std, 1.5) 
        normalized_current = centered_current / (effective_std + 1e-6)

        # Project with PCA
        raw_coords = state.pca.transform(normalized_current) # [K, 3]
        
        # TANH Bounding + DAMPING
        # Squeeze into roughly [-1, 1] range. 
        # Added 0.5 damping factor to keep them in the linear "middle" region.
        bounded_coords = np.tanh(raw_coords * 0.5) 

        # Decorate frame
        frame_agents = []
        for k in range(params.K):
            obj_name = world.objective_modes[obj_indices[k]]
            # Clean text: remove special tokens and artifacts
            # Remove <|...|> structural tags
            raw_text = thought_texts[k]
            clean_text = re.sub(r'<\|.*?\|>', '', raw_text)
            clean_text = clean_text.replace('Ġ', ' ').replace('Ċ', ' ').strip()
            
            # Scale & Decorate
            # Scale to [-50, 50] volume (Inside the new 100x100 box)
            pos_np = bounded_coords[k] * 50.0
            
            if k == 0 and t < 5 and params.test_mode:
                print(f"DEBUG: Step {t} | Agent 0 Pos: {pos_np} | Obj: {obj_name}")
            
            # Sanitize check
            if np.isnan(pos_np).any() or np.isinf(pos_np).any():
                 pos = [0.0, 0.0, 0.0]
                 print(f"WARNING: NaN/Inf detected for Agent {k} at Step {t}")
            else:
                 pos = pos_np.tolist()

            frame_agents.append({
                "id": k,
                "pos": pos,
                "color": OBJECTIVE_COLORS.get(obj_name, "#FFFFFF"),
                "token": clean_text,
                "objective": obj_name,
                "is_best": (k == best_agent_idx)
            })

        frame = {"step": t, "agents": frame_agents}
        state.frame_buffer.append(frame)
        yield frame
        await asyncio.sleep(0.01)

    # --- END OF EPISODE: OPTIMIZATION ---
    full_trace = torch.cat(trace_z, dim=1)
    rewards = StateObjectives.get_reward(full_trace, None, world.objective_modes, obj_indices)
    mean_kl = torch.stack(kl_penalties, dim=1).mean(dim=1)
    
    total_rewards = rewards - (0.5 * mean_kl)
    total_rewards = torch.clamp(total_rewards, min=-50.0, max=50.0)

    # Gradient update
    trajectory_log_probs = torch.cat(log_probs_actions, dim=1).sum(dim=1)
    if total_rewards.std() > 1e-6:
        adv = (total_rewards - total_rewards.mean()) / (total_rewards.std() + 1e-6)
    else:
        adv = total_rewards
    
    loss = -(adv * trajectory_log_probs).mean()
    
    if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world.root_agent.parameters(), 0.5)
        state.optimizer.step()
        
    state.episode_count += 1
    
    print(f"\nEpisode {state.episode_count} Complete")
    print(f"Avg Reward: {total_rewards.mean().item():.4f} | Avg KL: {mean_kl.mean().item():.4f} | Loss: {loss.item():.4f}")
    
    # Yield completion event
    sorted_indices = torch.argsort(total_rewards, descending=True)
    leaderboard = []
    for idx in sorted_indices:
        i = idx.item()
        leaderboard.append({
            "agent_id": i,
            "reward": total_rewards[i].item(),
            "objective": world.objective_modes[obj_indices[i]]
        })

    state.running = False
    yield {
        "event": "complete", 
        "stats": {
            "reward": total_rewards.mean().item(), 
            "kl": mean_kl.mean().item(),
            "winner_id": sorted_indices[0].item(),
            "winner_reward": total_rewards[sorted_indices[0]].item(),
            "leaderboard": leaderboard
        }
    }
