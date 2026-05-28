import numpy as np
import pytest

from competition.track_scene import build_track_model, resolve_go2_asset_model_dir
from go2_pg_env.track import StandardOvalTrack
from run_track_bonus import _validate_checkpoint
from track_bonus.controller_interface import (
    LOWLEVEL_ACTION_SIZE,
    LOWLEVEL_STATE_OBS_SIZE,
    TRACK_OBS_FEATURE_NAMES,
    build_track_controller_observation,
    validate_high_level_command,
)


def test_validate_high_level_command_keeps_values_and_validates_shape() -> None:
    command = validate_high_level_command(np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(command, np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, 0.2], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, np.nan, 0.2], dtype=np.float32))


def test_track_controller_observation_is_compact_track_state() -> None:
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(0.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[2] = 0.31
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, track=track)
    assert TRACK_OBS_FEATURE_NAMES == (
        "lap_fraction",
        "lateral_error_norm",
        "boundary_margin_norm",
        "heading_error_rad",
        "curvature_norm",
    )
    assert obs.as_array().shape == (5,)
    assert abs(obs.lateral_error_norm) < 1e-6
    assert obs.boundary_margin_norm == pytest.approx(1.0)
    assert abs(obs.heading_error_rad) < 1e-6
    assert 0.0 <= obs.lap_fraction < 1.0


def test_track_scene_compiles_single_dog_when_assets_are_available() -> None:
    try:
        resolve_go2_asset_model_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model = build_track_model(num_dogs=1, colors=["#2563EB"])
    assert model.nq == 19
    assert model.nu == 12


def test_student_track_scene_rejects_multi_dog_rendering() -> None:
    with pytest.raises(ValueError, match="one Go2"):
        build_track_model(num_dogs=2)


def test_checkpoint_validation_requires_hw1_actor_contract(tmp_path) -> None:
    config_path = tmp_path / "ppo_network_config.json"
    config_path.write_text(
        (
            '{"action_size": 12, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            '"observation_size": {"state": {"shape": [48]}}}'
        ),
        encoding="utf-8",
    )
    _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            '{"action_size": 12, '
            '"network_factory_kwargs": {"policy_obs_key": "privileged_state"}, '
            '"observation_size": {"state": {"shape": [48]}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy_obs_key"):
        _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            f'{{"action_size": {LOWLEVEL_ACTION_SIZE}, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            f'"observation_size": {{"state": {{"shape": [{LOWLEVEL_STATE_OBS_SIZE + 1}]}}}}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="state observation shape"):
        _validate_checkpoint(tmp_path)

    config_path.write_text(
        (
            f'{{"action_size": {LOWLEVEL_ACTION_SIZE + 1}, '
            '"network_factory_kwargs": {"policy_obs_key": "state"}, '
            f'"observation_size": {{"state": {{"shape": [{LOWLEVEL_STATE_OBS_SIZE}]}}}}}}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action_size"):
        _validate_checkpoint(tmp_path)
