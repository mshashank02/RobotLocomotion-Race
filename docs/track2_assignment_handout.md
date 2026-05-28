# Track 2 Handout

## Routes

1. Do your proposal-based final project.
2. Enter the Go2 oval-track tournament.
3. Do both, for bonus.

## Tournament Goal

Run Go2 around a 200 m oval track. Stay upright, stay inside the lane, and
maximize progress. A full lap is ranked by finish time; incomplete runs are
ranked by valid distance before fall or boundary exit.

## Controller

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

See `docs/controller_interface.md`.

## Starter Baseline

This repo gives:

- HW1-style Go2 PPO training code
- weak starter high-level planner
- track geometry and evaluation scripts
- Colab template
- 10-dog rollout renderer

It does not give a successful checkpoint or solved planner.

## What To Improve

- Low-level turning and command tracking.
- High-level track controller.
- Curriculum, reward, search, or RL training method.

## Tournament Outputs

```text
results.json
leaderboard.csv
race_rollouts.npz
race.mp4
```

Main ranking fields:

- `lap_completion`
- `valid_distance_m`
- `finish_time`
- `fall`
- `boundary_violation`

## Submission

```text
best_checkpoint/
planner_config.json or planner file
submission.json
track_eval/results.json
optional track_eval/race.mp4
short_report.pdf
```

## Notes

- Do not hard-code evaluator outputs.
- Do not change required metric names.
- Do not use privileged low-level actor observations.
- Use compact track-coordinate features for the high-level controller.
- Real robots would need localization to provide track coordinates.
