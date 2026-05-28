# High-Level Optimization Guide

The starter high-level planner is weak on purpose.

## Interface

```text
[lap_fraction, lateral_error_norm, boundary_margin_norm,
 heading_error_rad, curvature_norm]
  -> [vx, vy, yaw_rate]
```

The evaluator checks shape and finite values only. It does not clip commands.

## Starter Search

```bash
python train_highlevel_starter.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --output-dir artifacts/highlevel_train \
  --iterations 8 \
  --population 12
```

This is black-box search over starter planner parameters. It optimizes
`scores.composite_score` from `run_track_bonus.py`.

## Things To Try

- Tune `speed_mps`, `k_heading`, `k_lateral`, and command scaling inside the
  starter planner.
- Replace `track_bonus/planner.py` with an MLP or RL controller.
- Train the low-level policy to track nonzero `vy` and `yaw_rate`.
- Use staged evaluation: straight, turn entry, turn middle, turn exit, full lap.

## Watch These Metrics

- Low `lap_completion`: not enough progress.
- Low `valid_distance_m`: early fall, boundary exit, or very slow policy.
- `fall = true`: low-level instability or aggressive commands.
- `boundary_violation = true`: weak high-level line keeping.
- High lateral error: controller is safe but inaccurate.
- High slip or energy: inefficient turning.

## Minimal Loop

1. Run starter eval.
2. Inspect `results.json` and `race.mp4`.
3. Improve low-level tracking or high-level planner.
4. Re-evaluate.
5. Save best checkpoint, planner, metrics, and report.
