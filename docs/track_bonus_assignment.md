# Track 2 Bonus Project: 200 m Go2 Oval

## Goal

Build a hierarchical controller that makes Unitree Go2 run around a 200 m
standard oval track in MuJoCo:

```text
track coordinates -> high-level command [vx, vy, yaw_rate]
proprioception + command -> low-level Go2 locomotion policy
```

The low-level policy is the same Brax PPO checkpoint format used in HW1. The
bonus project adds the track geometry, high-level interface, evaluation, and
visualization.

## What You Start With

- HW1 Go2 joystick locomotion training code.
- A weak starter high-level planner in `configs/starter_planner.json`.
- `run_track_bonus.py` to evaluate a low-level checkpoint plus a high-level
  planner.
- `train_highlevel_starter.py`, a small black-box search scaffold that tunes
  the starter planner parameters.

The starter planner is not meant to solve the project. It is a readable
example of the interface you should improve.

## What You May Change

Recommended files:

- `go2_pg_env/joystick.py` for low-level command tracking improvements.
- `configs/course_config.json` for low-level curriculum and rewards.
- `track_bonus/planner.py` or a new high-level planner module.
- `train_highlevel_starter.py` or your own high-level training code.
- `configs/starter_planner.json` or your submitted planner config.

Do not change the evaluation metrics or output field names in a way that makes
your results incomparable.

## Track Observation

The high-level planner may use global track information, including:

- robot `x, y` position
- progress along the track centerline
- lateral error from the centerline
- distance to the track boundary
- heading error relative to the track tangent
- local curvature or lookahead heading

The low-level PPO policy should still receive the normal HW1 observation:
proprioception plus the high-level command.

## Evaluation Metrics

The benchmark reports:

- `lap_completion`
- `finish_time`
- `mean_progress_speed`
- `rms_lateral_error`
- `max_lateral_error`
- `min_boundary_margin_m`
- `fall`
- `boundary_violation`
- `energy_proxy`
- `foot_slip_proxy`

Completing the lap matters most, but faster and cleaner laps score higher.

## Submission

Submit:

- `best_checkpoint/`
- `planner_config.json` or your planner file
- `submission.json`
- `track_eval/results.json`
- optional `track_eval/race.mp4`
- a short report describing your low-level changes, high-level design,
  training method, metrics, and at least one failed idea.
