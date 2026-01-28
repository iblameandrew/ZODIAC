# Task Checklist: ZODIAC Agent Simulation

## 1. GRPO Loop Stabilization (Completed)
- [x] Knowledge Integration Backpass (Centroid Projection)
- [x] Semantic Backbone Anchoring (KL-Penalty)
- [x] Numerical Hardening (Gradient Clipping, Latent Norm, Temperature Floor)
- [x] Compositional Fusion (Sampled Output)

## 2. Coherence Tuning (Completed)
- [x] **Logit Fusion**: `L_final = L_base + α * L_agent` (Tethering)
- [x] **Top-K Filtering**: `k=50` to prevent tail collapse
- [x] **Dynamic Steering**: Ramp-up `α` from 0.0 to 0.4
- [x] **Repetition Penalty**: `-2.0` for recurring tokens

## 3. Semantic Polarity & Geometry (Completed)
- [x] **Zodiac Modes**: Define Expanders (Fire/Air) vs Reducers (Earth/Water)
- [x] **Semantic Re-Ranker**: Boost tokens based on distance/similarity to agent state
- [x] **Harmonic Correction**: Geometric steering of latent vectors (Push/Pull)
- [x] **Float32 Stability**: Fix CUDA asserts in distance metrics

## 4. Evolutionary Dynamics (Completed)
- [x] **Persistent Population**: State carried over across episodes via `population_states`
- [x] **Reproduction**: Survival of the fittest (Top 50%) -> Mutation -> Offspring
- [x] **Feedback Loop**: Evolution influences `forward_agent` context
- [x] **Lineage Tracking**: Visual tags in logs (`GEN-0`, `CHILD-EpX`)

## 5. Verification
- [x] Verify simulation runs without crash (Exit Code 0)
- [x] Verify coherent output generation
- [x] Verify visual logging of Evolution and Polarity

## 6. Deep Inference Refactor (Completed)
- [x] **Backbone Restoration**: Implement `world.backbone.model(inputs_embeds=curr_emb)` for true hidden states.
- [x] **Residual Steering**: Convert Agent to Delta-Z mechanism (`z_steered = z + alpha * delta`).
- [x] **Stabilized Arithmetic**: Normalize steering bias using logit standard deviation.
- [x] **Vectorized History**: Fix `curr_emb` concatenations for deep pass.

## 7. Entropy-Gated Steering (Completed)
- [x] **Delete Semantic Re-Ranker**: Remove flawed logit manipulation.
- [x] **Entropy Gate**: Implement `sigmoid(entropy - 3.0)` to steer only on high uncertainty.
- [x] **Coherence Guardrail**: Dampen steering if KL > 1.5.
- [x] **Strength Cap**: Reduce max steering strength to 0.15.

## 8. Stabilization (Completed)
- [x] **Steering Normalization**: Force `thought_delta` to unit vector.
- [x] **Logit Blending**: `final = (1-a)*base + a*steered` for grammar.
- [x] **Top-K Filtering**: Truncated junk tail $(\text{k=40})$.
- [x] **Numerical Hardening**: `nan_to_num` sanitization across all layers.
