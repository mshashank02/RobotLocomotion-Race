# Track 2 Requirements

Goal: run Go2 as far as possible around a 200 m oval track in MuJoCo.

## Routes

- Proposal-based final project.
- Go2 oval-track tournament.
- Both, for bonus.

## Tournament Controller

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

Use the interface in `docs/controller_interface.md`. The low-level checkpoint
should stay compatible with the HW1 Brax PPO format.

## Allowed

- Reuse a HW1 checkpoint.
- Retrain or modify the low-level Go2 policy.
- Replace or train the high-level controller.
- Add your own configs, scripts, planner modules, or report artifacts.

## Not Allowed

- Hard-code benchmark results.
- Delete or rename required output fields.
- Bypass the low-level policy with prewritten joint trajectories.
- Use privileged low-level actor observations beyond normal `state`.

## Commands

Train:

```bash
python train.py \
  --config configs/course_config.json \
  --stage both \
  --output-dir artifacts/low_level_train
```

Evaluate:

```bash
python run_track_bonus.py \
  --checkpoint-dir artifacts/low_level_train/best_checkpoint \
  --planner-config configs/starter_planner.json \
  --output-dir artifacts/track_eval
```

## Ranking

- Completed laps rank before incomplete runs.
- Completed laps rank by lower `finish_time`.
- Incomplete runs rank by higher `valid_distance_m`.
- Ties use failures, boundary margin, lateral error, slip, and energy.

## Outputs And Metrics

Outputs: `results.json`, `leaderboard.csv`, `race_rollouts.npz`, optional
`race.mp4`.

Main metrics: `lap_completion`, `valid_distance_m`, `finish_time`, `fall`,
`boundary_violation`.

## Submission

```text
best_checkpoint/
planner_config.json or planner file
submission.json
track_eval/results.json
optional track_eval/race.mp4
short_report.pdf
```

Report briefly: low-level changes, high-level design, training/search method,
final metrics, one failed idea, and real-robot localization assumptions.

## Grading

Proposal projects use the final-project rubric. Tournament entries use
leaderboard performance, method quality, analysis, presentation, and
reproducibility. Doing both routes is eligible for bonus.
