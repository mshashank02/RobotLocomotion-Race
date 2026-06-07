"""Starter high-level planner for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    learned_weights_path: str | None = None
    residual_scales: tuple[float, float, float] = (0.35, 0.18, 0.45)
    min_vx_mps: float = 0.0
    max_vx_mps: float = 1.2
    min_vy_mps: float = -0.4
    max_vy_mps: float = 0.4
    min_yaw_rate_radps: float = -1.0
    max_command_yaw_rate_radps: float = 1.0
    track_length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        if "residual_scales" in values:
            values["residual_scales"] = tuple(float(value) for value in values["residual_scales"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
            "learned_weights_path": self.learned_weights_path,
            "residual_scales": list(self.residual_scales),
            "min_vx_mps": self.min_vx_mps,
            "max_vx_mps": self.max_vx_mps,
            "min_vy_mps": self.min_vy_mps,
            "max_vy_mps": self.max_vy_mps,
            "min_yaw_rate_radps": self.min_yaw_rate_radps,
            "max_command_yaw_rate_radps": self.max_command_yaw_rate_radps,
            "track_length_m": self.track_length_m,
            "turn_radius_m": self.turn_radius_m,
            "half_width_m": self.half_width_m,
        }


def make_zero_residual_weights() -> dict[str, np.ndarray]:
    """Return MLP parameters for a zero residual policy."""
    return {
        "w1": np.zeros((5, 64), dtype=np.float32),
        "b1": np.zeros(64, dtype=np.float32),
        "w2": np.zeros((64, 64), dtype=np.float32),
        "b2": np.zeros(64, dtype=np.float32),
        "w3": np.zeros((64, 3), dtype=np.float32),
        "b3": np.zeros(3, dtype=np.float32),
    }


def load_residual_weights(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    weights = {key: np.asarray(payload[key], dtype=np.float32) for key in ("w1", "b1", "w2", "b2", "w3", "b3")}
    expected_shapes = {
        "w1": (5, 64),
        "b1": (64,),
        "w2": (64, 64),
        "b2": (64,),
        "w3": (64, 3),
        "b3": (3,),
    }
    for key, shape in expected_shapes.items():
        if weights[key].shape != shape:
            raise ValueError(f"Residual weight {key!r} must have shape {shape}, got {weights[key].shape}.")
    return weights


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(self, config: StarterPlannerConfig, residual_weights: dict[str, np.ndarray] | None = None) -> None:
        if config.planner_type not in {"starter_pd", "residual_mlp"}:
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()
        self._residual_weights = residual_weights
        if self.config.planner_type == "residual_mlp" and self._residual_weights is None:
            if self.config.learned_weights_path is None:
                self._residual_weights = make_zero_residual_weights()
            else:
                self._residual_weights = load_residual_weights(Path(self.config.learned_weights_path))

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        config = StarterPlannerConfig.load(path)
        if config.learned_weights_path is None:
            return cls(config)
        weights_path = Path(config.learned_weights_path)
        if not weights_path.is_absolute():
            weights_path = path.resolve().parent / weights_path
        return cls(config, residual_weights=load_residual_weights(weights_path))

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        base_command = self.base_command_from_observation(obs)
        if self.config.planner_type == "starter_pd":
            return base_command
        delta = self._residual_from_observation(obs)
        command = base_command + delta
        return np.asarray(
            [
                np.clip(command[0], self.config.min_vx_mps, self.config.max_vx_mps),
                np.clip(command[1], self.config.min_vy_mps, self.config.max_vy_mps),
                np.clip(command[2], self.config.min_yaw_rate_radps, self.config.max_command_yaw_rate_radps),
            ],
            dtype=np.float32,
        )

    def base_command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)

    def _residual_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        if self._residual_weights is None:
            return np.zeros(3, dtype=np.float32)
        x = obs.as_array().astype(np.float32)
        # Normalize the official 5D observation while preserving the evaluator interface.
        x = x.copy()
        x[0] = 2.0 * x[0] - 1.0
        x[1] = np.clip(x[1], -2.0, 2.0)
        x[2] = np.clip(x[2], -1.0, 1.0)
        x[3] = np.clip(x[3] / math.pi, -1.0, 1.0)
        x[4] = np.clip(x[4], -1.0, 1.0)
        w = self._residual_weights
        h1 = np.tanh(x @ w["w1"] + w["b1"])
        h2 = np.tanh(h1 @ w["w2"] + w["b2"])
        raw = np.tanh(h2 @ w["w3"] + w["b3"])
        return (raw * np.asarray(self.config.residual_scales, dtype=np.float32)).astype(np.float32)
