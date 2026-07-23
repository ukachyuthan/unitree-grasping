# Research Notes — GraspNet-Bootstrapped RL with Procedural Objects

**Paper title (working):** "GraspNet-Bootstrapped RL with Procedural Objects for Generalizable Grasping"
**Target:** arXiv + LinkedIn post
**Key claim:** A PointNet-based RL policy trained only on procedurally generated shapes grasps real, never-seen objects zero-shot.

---

## 2026-06-28 / 2026-06-29 — Project bootstrap + first training run

### Project pivot
- Deleted all G1 walking code (SAC trainer, replay buffer, terrain configs, walking envs, play/visualize scripts).
- Project is now purely grasping. Branch: `feat-pivot-to-grasping`.

### Architecture decisions (paper-relevant)

**Robot-agnostic policy interface**
- Observation = [object point cloud in robot-base frame | EE pose | goal pose]. No joint names or DOF counts visible to the policy.
- Any robot with IK + calibrated camera (T_cam_robot) can run the trained weights unchanged.
- Camera-to-robot transform applied in `pointcloud_utils.transform_pointcloud()` — same function works in sim and on a real RealSense.

**PointNet encoder**
- `(B, N, 3) → (B, embed_dim=128)` via Conv1d shared MLP + symmetric max-pool.
- No T-Net (unnecessary overhead for fixed overhead camera geometry).
- Trainable end-to-end through PPO gradients.
- Actor and critic have independent PointNet weights (mirrors standard RSL-RL ActorCritic design).

**GT point cloud synthesis during training**
- Chose GT surface sampling + 3 mm Gaussian noise over a simulated depth camera.
- Reason: TiledCamera + `--enable_cameras` triggers the RTX/Hydra renderer which crashes on this machine (IOMMU + RTX 3060 Ti driver incompatibility — TLAS buffer overflow during headless startup).
- Justification for paper: same approach as DexPoint, UniDexGrasp, M2T2. Training with GT PCs + noise augmentation avoids sim rendering artifacts and generalises better than noisy rendered depth.
- Real camera path (`depth_to_pointcloud_world()` in `pointcloud_utils.py`) is preserved for inference/eval.

**Action space**
- Joint-space mode: Δjoint(5 arm DOF) + gripper(1), padded to 7 dims. Used for initial training (simpler).
- EE-space mode: Δpos(3) + Δrot_axisangle(3) + gripper(1) → Jacobian-transpose IK. Robot-agnostic at deployment.
- Controlled by `use_joint_space_control` flag in cfg (default True).

### Procedural object generator (`scripts/generate_objects.py`)
13 shape families, chosen specifically to break policies trained only on primitives:
1. Superquadric (covers box→sphere→cylinder continuously)
2. Torus — centroid not on surface, challenges grasp centering
3. L-shape
4. T-shape
5. C-shape (open arc)
6. Dumbbell — bimodal mass distribution
7. Wedge/triangular prism
8. Star prism — sharp irregular boundary
9. Bracket/frame — rectangular through-hole
10. Stepped cylinder
11. Twisted bar — helical cross-section
12. Irregular polygon extrusion (non-convex boundary)
13. Random convex hull from ellipsoidal point cloud

**Bug fixed today:** `make_twisted_bar` was replacing the twisted mesh with its convex hull (line overwrite). `make_irregular_extrusion` was convex-hulling the polygon before extrusion. Both fixed — shapes now actually non-convex.

**Zero-shot eval protocol:** train seed=0, eval seed=99 → completely different shape instances across all 13 families.

### Hardware / infra
- GPU: RTX 3060 Ti (8 GB VRAM)
- CPU: AMD Ryzen 5 5600X
- OS: Ubuntu 22.04, driver 595.71.05
- Isaac Lab 4.5, rsl_rl (2.x API — no `actor_critic_class` arg to OnPolicyRunner)
- 256 envs, headless (no rendering), ~4400 steps/s

### Bugs fixed today (in order)
1. `ee_body_name = "left_rubber_hand"` → `"left_palm_link"` (body index 28, confirmed by inspect_g1_joints.py)
2. `effort_limit` → `effort_limit_sim`, `velocity_limit` → `velocity_limit_sim` (Isaac Lab 4.5 deprecation)
3. Joint name conflict in actuator cfg: `".*_five_joint"` + explicit `"left_five_joint"` in same group → removed explicit duplicates
4. TiledCamera requires `--enable_cameras` → caused RTX segfault → replaced with GT PC synthesis
5. `left_elbow_pitch_joint` default pose -1.0 rad outside joint limits [-0.227, 3.421] → changed to +1.0
6. `OnPolicyRunner` does not accept `actor_critic_class` kwarg in rsl_rl 2.x → create runner with standard `ActorCritic` placeholder, then swap `runner.alg.actor_critic` and reset `runner.alg.optimizer`
7. PPO config key `use_clipping` → `use_clipped_value_loss`
8. `use_joint_space_control` being overridden to `False` by default `--joint_space` flag (store_true defaults False) → fixed to only override when flag is explicitly passed
9. `_apply_ee_delta` Jacobian shape: Isaac Lab returns `(B, num_bodies * 6, num_dofs)`, not `(B, 6, num_dofs)` → slice EE body rows as `jac_full[:, ee_body_idx*6 : ee_body_idx*6+6, :]`
10. **Reward killed gradient signal**: place reward `-dist_obj_goal * 10` was always active even when object hadn't moved → constant dominant negative reward → near-zero advantages → zero surrogate loss → policy frozen at initialization. Fix: gate place reward on `lifted = (obj_z - table_z) > 0.05m`.

### First training run results (1500 iters, 256 envs, ~35 min)
- Mean reward: -993 (flat, no improvement)
- Entropy: 9.9326 (theoretical max for 7-dim Gaussian std=1.0 — policy never updated)
- Surrogate loss: ~0 (zero policy gradient — confirmed reward design bug)
- Root cause: place reward dominated advantages, masking approach gradient
- **Action:** reward fix applied, re-training needed

---

## Next sessions — planned work

### Phase 2 — Procedural object training (key paper contribution)
- Generate 1000 train meshes (seed=0), 200 eval meshes (seed=99)
- Convert OBJ → USD for Isaac Lab rigid body physics (use `MeshConverter`)
- Update env to load random USD mesh per episode instead of 3 hardcoded primitives
- Update `_synthesize_pointcloud` to sample from trimesh mesh vertices instead of analytical surface formulas
- Re-train with procedural objects

### Phase 3 — Zero-shot evaluation
- Train-seed policy evaluated on eval-seed objects
- Metric: success rate on train shapes vs novel shapes
- Target: comparable success rate → PointNet generalises through geometry

### Phase 4 — Ablations (for paper)
- MLP baseline (same obs, no PointNet) vs PointNet — expect MLP to drop on novel shapes
- 3 primitives only vs 13-family procedural — quantify benefit of shape diversity
- Noise level ablation: 0 mm / 3 mm / 10 mm sensor noise

### Phase 5 — Real demo (optional, strong for paper + LinkedIn)
- RealSense D435 depth → `depth_to_pointcloud_world()` → policy → G1 SDK joint commands
- Requires hand-eye calibration (T_cam_robot)
- Can demo on objects never seen in training or sim
