# Controller Interface

## Architecture

```text
high-level:
  5D track observation -> [vx, vy, yaw_rate]

low-level:
  proprioception + command -> 12 Go2 joint actions
```

## High-Level Input

```text
[
  lap_fraction,
  lateral_error_norm,
  boundary_margin_norm,
  heading_error_rad,
  curvature_norm
]
```

Defined in `track_bonus/controller_interface.py`.

## High-Level Output

```text
[vx_mps, vy_mps, yaw_rate_radps]
```

Shape must be `(3,)`. Values must be finite. The evaluator does not clip or
rescale commands.

## Low-Level Checkpoint

Use the HW1 Brax PPO format:

- `ppo_network_config.json` exists
- actor `policy_obs_key = "state"`
- no `privileged_state` actor
- 12-dimensional action

## Tournament Rendering

Scoring runs each entry independently. Visualization combines saved rollouts:

```json
{
  "entries": [
    {
      "name": "team_a",
      "rollout_npz": "team_a/track_eval/race_rollouts.npz",
      "color": "#2563EB"
    }
  ]
}
```

```bash
python scripts/render_track_tournament.py \
  --entries tournament_entries.json \
  --visual-lane-offsets \
  --output-dir artifacts/tournament_render
```

Maximum entries: 10.
