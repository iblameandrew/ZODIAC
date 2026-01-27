# Walkthrough: Zodiac Agent Simulation (Stable & Evolutionary)

The simulation has been upgraded to a **Hybrid Evolutionary System** with Semantic Geometry. It successfully stabilizes the "Grpo Loop" by tethering agents to the Llama-3 backbone while implementing harmonic geometric constraints.

## System Architecture

### 1. Semantic Logit Fusion (The "Tether")
To prevents mode collapse into random characters ("!!!!!"), we combine the frozen backbone's grammar with the agent's intent.
- **Formula**: `Logits_Final = Logits_Backbone + (0.4 * Logits_Agent)`
- **Mechanism**: Dynamic steering strength (ramps up from 0.0 to 0.4) ensures proper sentence initiation.
- **Top-K Filtering**: Only the top 50 tokens are considered to prevent "tail" sampling.
- **Repetition Penalty**: `-2.0` penalty for previously generated tokens.

### 2. Semantic Polarity & Harmonic Correction
Agents are assigned a semantic "polarity" based on their Zodiac mode, enforcing geometric structural integrity in the latent space.
- **EXPANDERS** (Fire/Air): Latent vectors are **pushed away** from the population centroid (Exploration).
- **REDUCERS** (Earth/Water): Latent vectors are **pulled towards** the population centroid (Consolidation).
- **Output**: Visualized in logs as `[EXPAND]` or `[REDUCE]`.

### 3. Evolutionary Population
Instead of resetting agents every episode, the system now maintains a persistent population (`population_states`).
- **Natural Selection**: At the end of each episode, agents are ranked by reward.
- **Reproduction**: The top 50% spawn mutated offspring (`noise * 0.1`) to replace the bottom 50%.
- **Inheritance**: The persistent population state is injected as "Neighbor Context" into the `GranularAgent.think` process.

### 4. Technical Stability
- **Float32 Precision**: Distance metrics and norm calculations use Float32 to prevent Float16 overflow/underflow (fixing `device-side assert` crashes).
- **Division Safety**: Norm denominators are clamped (`max(norm, 1e-6)`) to prevent division-by-zero crashes.
- **Sanitization**: Inputs to sensitive ops like `cosine_similarity` are sanitized (`nan_to_num`).

## Verification Results

The simulation runs stably (Exit Code 0).
- **Text Coherence**: Output has shifted from garbage to semi-coherent thematic words (e.g., "Observation", "Character").
- **Visual Logs**: 
    ```
    Iteration 01 | Avg Reward: -0.0033 | Avg KL: 0.0045
       [0]* Mode: KINETIC_VELOCITY          [EXPAND] | Reward: -0.0029 | " motion..."
    ...
    >> Evolution: Agents [2, 0] spawning children...
    ```
