# Controller Interface

High-level controllers must use this interface so different submissions can be
evaluated together.

```text
5D track observation -> [vx, vy, yaw_rate] -> Go2 low-level policy
```

## Input

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

## Output

```text
[vx_mps, vy_mps, yaw_rate_radps]
```

Shape must be `(3,)`. Values must be finite. The evaluator does not clip or
rescale commands.

## Checkpoint

Use the HW1 Brax PPO format:

- `ppo_network_config.json` exists
- actor `policy_obs_key = "state"`
- `state` observation shape is 48
- no `privileged_state` actor
- 12-dimensional action

## Multi-Dog Render

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
