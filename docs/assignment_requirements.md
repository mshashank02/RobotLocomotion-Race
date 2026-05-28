# Track 2 Requirements

Goal: run Go2 as far as possible around a 200 m oval track in MuJoCo.

## Options

- Proposal-based final project.
- Go2 oval-track tournament.
- Both, for bonus.

## Tournament

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

Use `docs/controller_interface.md`. The low-level checkpoint should stay
compatible with the HW1 Brax PPO format.

## Allowed

- Reuse a HW1 checkpoint.
- Retrain or modify the low-level Go2 policy.
- Replace or train the high-level controller.

## Not Allowed

- Hard-code benchmark results.
- Delete or rename required output fields.
- Bypass the low-level policy with prewritten joint trajectories.
- Use privileged actor observations beyond normal `state`.

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
planner_config.json and changed planner code
submission.json
track_eval/results.json
optional track_eval/race.mp4
short_report.pdf
```

Report briefly: low-level changes, high-level design, training/search method,
final metrics, and one failed idea.

## Grading

Proposal projects use the final-project rubric. Tournament entries use the
same distribution as the final-project route. Ranking uses track distance,
completion time, method quality, analysis, presentation, and reproducibility.
Doing both routes is eligible for bonus.
