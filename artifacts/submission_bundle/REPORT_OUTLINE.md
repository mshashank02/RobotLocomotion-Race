# EEC289A Go2 Oval Track Report Outline

Keep final report <= 8 pages.

## 1. Summary
- Alternative Track 2 leaderboard submission.
- Low-level controller: frozen Brax PPO velocity tracker.
- High-level controller: learned residual MLP planner.

## 2. Low-Level Locomotion Policy
- Checkpoint: artifacts/experiments/stage2_adaptive_curriculum_reward_v5/best_checkpoint
- Interface: commanded [vx, vy, yaw_rate].
- Actor observation: normal state observation only, no privileged actor observations.
- Training notes: stage 1 forward motion, stage 2 adaptive command curriculum.

## 3. High-Level Planner
- Official input: [lap_fraction, lateral_error_norm, boundary_margin_norm, heading_error_rad, curvature_norm].
- Output: [vx_mps, vy_mps, yaw_rate_radps].
- Architecture: analytic baseline plus residual MLP 5 -> 64 -> 64 -> 3, tanh activations.
- Learned files: artifacts/residual_planner_train_v2/planner_config.json and residual_weights.npz.

## 4. Training Objective
- Reward encouraged forward progress, lap completion, speed, boundary safety, fall avoidance, lateral centering, and smooth commands.
- Residual planner optimized with black-box rollouts using the frozen low-level policy.

## 5. Final Evaluation
- Lap completion: 1.0
- Finish time: 212.2 s
- Fall: false
- Boundary violation: false
- Composite score: 0.9767

## 6. Reproducibility
- Run run_track_bonus.py with the final checkpoint and planner config listed in submission.json.
- Include code files: track_bonus/planner.py, train_residual_planner.py, run_track_bonus.py, train.py, configs/server_runtime_config.json.
