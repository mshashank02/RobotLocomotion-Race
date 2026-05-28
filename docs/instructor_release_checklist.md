# Instructor Release Checklist

## Must Exist

- `README.md`
- `docs/track2_assignment_handout.md`
- `docs/assignment_requirements.md`
- `docs/controller_interface.md`
- `docs/high_level_optimization_guide.md`
- `notebooks/track_bonus_colab_template.ipynb`

## Must Not Include

- trained checkpoints
- optimized planners
- teacher rollouts
- solution videos
- private outputs
- grading logs

## Smoke Tests

```bash
python -m pytest -q tests
python -m py_compile \
  run_track_bonus.py \
  train_highlevel_starter.py \
  scripts/render_track_tournament.py \
  track_bonus/controller_interface.py \
  track_bonus/planner.py \
  track_bonus/scoring.py \
  go2_pg_env/track.py
```

Notebook check:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

nb = nbformat.read(Path("notebooks/track_bonus_colab_template.ipynb"), as_version=4)
assert any("track2_assignment_handout" in cell.source for cell in nb.cells)
assert any("controller_interface" in cell.source for cell in nb.cells)
assert any("CHECKPOINT_DIR does not exist" in cell.source for cell in nb.cells)
print("notebook ok")
PY
```

10-dog compile check, after assets are available:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --output-dir artifacts/ten_dog_compile_check \
  --no-render
```

Expected: `model_nq = 190`, `model_nu = 120`.
