#!/usr/bin/env python3
"""Train a residual MLP high-level planner for the fixed oval track.

The low-level Go2 policy is frozen.  This script optimizes only the high-level
residual network used by ``StarterTrackPlanner.load(...)``:

    analytic baseline command + residual_mlp(official_5d_observation)

The optimizer is a simple evolution-strategy loop.  It treats the simulator as a
black-box objective, which keeps the competition evaluator interface unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from course_common import DEFAULT_CONFIG_PATH, lazy_import_stack, load_json, set_runtime_env
from run_track_bonus import _make_env, _validate_checkpoint, rollout
from test_policy import load_policy_with_workaround
from track_bonus.official_track import official_track, official_track_config
from track_bonus.planner import StarterPlannerConfig, StarterTrackPlanner, load_residual_weights, make_zero_residual_weights
from track_bonus.scoring import compute_track_bonus_metrics, score_track_bonus


ROOT = Path(__file__).resolve().parent
WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Frozen low-level best_checkpoint.")
    parser.add_argument("--base-planner-config", type=Path, default=ROOT / "configs" / "starter_planner.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-name", choices=["stage_1", "stage_2"], default="stage_2")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--eval-seconds", type=float, default=300.0)
    parser.add_argument("--start-samples", type=int, default=1, help="Number of start positions per candidate.")
    parser.add_argument("--sigma", type=float, default=0.035, help="Initial parameter search stddev.")
    parser.add_argument("--min-sigma", type=float, default=0.006)
    parser.add_argument("--init-planner-config", type=Path, default=None, help="Existing residual planner config to fine-tune.")
    parser.add_argument("--base-speed", type=float, default=0.55, help="Analytic baseline speed used by the residual planner.")
    parser.add_argument("--residual-vx-scale", type=float, default=0.65)
    parser.add_argument("--residual-vy-scale", type=float, default=0.22)
    parser.add_argument("--residual-yaw-scale", type=float, default=0.65)
    parser.add_argument("--target-finish-seconds", type=float, default=260.0)
    parser.add_argument("--target-progress-speed", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def flatten_weights(weights: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(weights[key], dtype=np.float32).ravel() for key in WEIGHT_KEYS]).astype(np.float32)


def unflatten_weights(vector: np.ndarray, template: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float32)
    weights: dict[str, np.ndarray] = {}
    cursor = 0
    for key in WEIGHT_KEYS:
        shape = template[key].shape
        size = int(np.prod(shape))
        weights[key] = vector[cursor : cursor + size].reshape(shape).astype(np.float32)
        cursor += size
    if cursor != vector.size:
        raise ValueError(f"Unused parameter values: consumed {cursor}, vector has {vector.size}.")
    return weights


def save_weights(path: Path, weights: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(weights[key], dtype=np.float32) for key in WEIGHT_KEYS})


def command_stats(commands: np.ndarray) -> dict[str, Any]:
    commands = np.asarray(commands, dtype=np.float32)
    if commands.size == 0:
        return {}
    diffs = np.diff(commands, axis=0)
    return {
        "mean": commands.mean(axis=0).tolist(),
        "std": commands.std(axis=0).tolist(),
        "min": commands.min(axis=0).tolist(),
        "max": commands.max(axis=0).tolist(),
        "mean_abs_delta": np.mean(np.abs(diffs), axis=0).tolist() if len(diffs) else [0.0, 0.0, 0.0],
        "smoothness_cost": float(np.mean(np.sum(np.square(diffs), axis=1))) if len(diffs) else 0.0,
    }


def rollout_reward(
    metrics: dict[str, Any],
    commands: np.ndarray,
    *,
    track_length_m: float,
    eval_seconds: float,
    target_finish_seconds: float,
    target_progress_speed: float,
) -> tuple[float, dict[str, float]]:
    stats = command_stats(commands)
    smoothness = float(stats.get("smoothness_cost", 0.0))
    forward_progress = float(metrics["valid_distance_m"])
    boundary_penalty = 120.0 if metrics["boundary_violation"] else 0.0
    fall_penalty = 120.0 if metrics["fall"] else 0.0
    lateral_penalty = 6.0 * float(metrics["rms_lateral_error"]) + 2.0 * float(metrics["max_lateral_error"])
    smoothness_penalty = 8.0 * smoothness
    mean_progress_speed = float(metrics["mean_progress_speed"])
    finish_time = metrics["finish_time"]
    if finish_time is None:
        remaining_distance = max(0.0, float(track_length_m) - forward_progress)
        finish_bonus = 0.0
        unfinished_penalty = 80.0 + 1.25 * remaining_distance
    else:
        finish_bonus = 240.0 + 1.5 * max(0.0, float(target_finish_seconds) - float(finish_time))
        unfinished_penalty = 0.0
    speed_error = float(target_progress_speed) - mean_progress_speed
    speed_penalty = 130.0 * max(0.0, speed_error)
    speed_bonus = 70.0 * max(0.0, mean_progress_speed - float(target_progress_speed))
    reward = (
        forward_progress
        + finish_bonus
        + speed_bonus
        - unfinished_penalty
        - speed_penalty
        - boundary_penalty
        - fall_penalty
        - lateral_penalty
        - smoothness_penalty
    )
    terms = {
        "forward_progress": forward_progress,
        "finish_bonus": finish_bonus,
        "unfinished_penalty": unfinished_penalty,
        "speed_bonus": speed_bonus,
        "speed_penalty": speed_penalty,
        "boundary_violation_penalty": boundary_penalty,
        "fall_penalty": fall_penalty,
        "lateral_error_penalty": lateral_penalty,
        "command_smoothness_penalty": smoothness_penalty,
    }
    return float(reward), terms


def evaluate_candidate(
    *,
    vector: np.ndarray,
    template: dict[str, np.ndarray],
    base_config: StarterPlannerConfig,
    stack: dict[str, Any],
    env: Any,
    policy: Any,
    eval_steps: int,
    eval_seconds: float,
    starts: list[float],
    seed: int,
    force_cpu: bool,
    target_finish_seconds: float,
    target_progress_speed: float,
) -> dict[str, Any]:
    weights = unflatten_weights(vector, template)
    planner = StarterTrackPlanner(base_config, residual_weights=weights)
    track = official_track()
    episode_records = []
    rewards = []
    composite_scores = []
    for episode_idx, start_s in enumerate(starts):
        result = rollout(
            stack=stack,
            env=env,
            policy=policy,
            planner=planner,
            track=track,
            num_steps=eval_steps,
            seed=int(seed) + 1009 * episode_idx,
            start_s=float(start_s),
            force_cpu=force_cpu,
        )
        metrics = compute_track_bonus_metrics(result, track)
        scores = score_track_bonus(metrics)
        reward, reward_terms = rollout_reward(
            metrics,
            result["command"],
            track_length_m=float(track.length_m),
            eval_seconds=float(eval_seconds),
            target_finish_seconds=float(target_finish_seconds),
            target_progress_speed=float(target_progress_speed),
        )
        rewards.append(reward)
        composite_scores.append(float(scores["composite_score"]))
        episode_records.append(
            {
                "start_s_m": float(start_s),
                "reward": reward,
                "reward_terms": reward_terms,
                "metrics": metrics,
                "scores": scores,
                "command_stats": command_stats(result["command"]),
            }
        )
    return {
        "reward": float(np.mean(rewards)),
        "composite_score": float(np.mean(composite_scores)),
        "episodes": episode_records,
    }


def make_residual_config(
    base: StarterPlannerConfig,
    weights_path: str,
    *,
    base_speed: float,
    residual_scales: tuple[float, float, float],
) -> StarterPlannerConfig:
    values = base.to_dict()
    values["planner_type"] = "residual_mlp"
    values["learned_weights_path"] = weights_path
    values["speed_mps"] = float(base_speed)
    values["min_speed_mps"] = min(float(values.get("min_speed_mps", 0.12)), float(base_speed))
    values["residual_scales"] = list(residual_scales)
    values.update(official_track_config())
    return StarterPlannerConfig.from_dict(values)


def load_initial_weights(config_path: Path) -> tuple[StarterPlannerConfig, dict[str, np.ndarray]]:
    config = StarterPlannerConfig.load(config_path)
    if config.learned_weights_path is None:
        raise ValueError(f"{config_path} does not define learned_weights_path.")
    weights_path = Path(config.learned_weights_path)
    if not weights_path.is_absolute():
        weights_path = config_path.resolve().parent / weights_path
    return config, load_residual_weights(weights_path)


def main() -> None:
    args = parse_args()
    _validate_checkpoint(args.checkpoint_dir)
    if args.force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=bool(args.force_cpu))

    rng = np.random.default_rng(int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_log.jsonl"

    course_cfg = load_json(args.config)
    course_cfg["runtime_overrides"] = {}
    eval_steps = int(round(float(args.eval_seconds) / float(course_cfg["control"]["ctrl_dt"])))
    start_count = max(int(args.start_samples), 1)
    starts = [0.0] if start_count == 1 else np.linspace(0.0, official_track().length_m, start_count, endpoint=False).tolist()

    stack = lazy_import_stack()
    env = _make_env(stack, course_cfg, args.stage_name, episode_steps=eval_steps)
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not args.force_cpu:
        policy = stack["jax"].jit(policy)

    template = make_zero_residual_weights()
    if args.init_planner_config is None:
        base_config = StarterPlannerConfig.load(args.base_planner_config)
        initial_weights = template
    else:
        base_config, initial_weights = load_initial_weights(args.init_planner_config)

    residual_config = make_residual_config(
        base_config,
        "residual_weights.npz",
        base_speed=float(args.base_speed),
        residual_scales=(
            float(args.residual_vx_scale),
            float(args.residual_vy_scale),
            float(args.residual_yaw_scale),
        ),
    )
    mean = flatten_weights(initial_weights)
    sigma = np.full(mean.shape, float(args.sigma), dtype=np.float32)
    best_vector = mean.copy()
    best_eval = evaluate_candidate(
        vector=best_vector,
        template=template,
        base_config=residual_config,
        stack=stack,
        env=env,
        policy=policy,
        eval_steps=eval_steps,
        eval_seconds=float(args.eval_seconds),
        starts=starts,
        seed=int(args.seed),
        force_cpu=bool(args.force_cpu),
        target_finish_seconds=float(args.target_finish_seconds),
        target_progress_speed=float(args.target_progress_speed),
    )
    best_reward = float(best_eval["reward"])
    save_weights(output_dir / "residual_weights.npz", unflatten_weights(best_vector, template))
    save_json(output_dir / "planner_config.json", residual_config.to_dict())
    save_json(output_dir / "best_eval.json", best_eval)

    elite_count = max(1, int(round(float(args.population) * float(args.elite_frac))))
    with history_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"iteration": -1, "candidate": 0, "is_best": True, **best_eval}) + "\n")

        for iteration in range(int(args.iterations)):
            candidates = []
            noise = rng.normal(0.0, 1.0, size=(int(args.population), mean.size)).astype(np.float32)
            noise[0] = 0.0
            vectors = mean[None, :] + noise * sigma[None, :]
            for candidate_idx, vector in enumerate(vectors):
                evaluation = evaluate_candidate(
                    vector=vector,
                    template=template,
                    base_config=residual_config,
                    stack=stack,
                    env=env,
                    policy=policy,
                    eval_steps=eval_steps,
                    eval_seconds=float(args.eval_seconds),
                    starts=starts,
                    seed=int(args.seed) + 100_000 * iteration + candidate_idx,
                    force_cpu=bool(args.force_cpu),
                    target_finish_seconds=float(args.target_finish_seconds),
                    target_progress_speed=float(args.target_progress_speed),
                )
                record = {
                    "iteration": iteration,
                    "candidate": candidate_idx,
                    "reward": evaluation["reward"],
                    "composite_score": evaluation["composite_score"],
                    "episodes": evaluation["episodes"],
                }
                candidates.append((float(evaluation["reward"]), vector.copy(), record, evaluation))
                if float(evaluation["reward"]) > best_reward:
                    best_reward = float(evaluation["reward"])
                    best_vector = vector.copy()
                    save_weights(output_dir / "residual_weights.npz", unflatten_weights(best_vector, template))
                    save_json(output_dir / "planner_config.json", residual_config.to_dict())
                    save_json(output_dir / "best_eval.json", evaluation)
                    record["is_best"] = True
                else:
                    record["is_best"] = False
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(
                    f"iter={iteration} cand={candidate_idx} reward={evaluation['reward']:.3f} "
                    f"score={evaluation['composite_score']:.3f} best={best_reward:.3f}",
                    flush=True,
                )

            candidates.sort(key=lambda item: item[0], reverse=True)
            elite_vectors = np.stack([item[1] for item in candidates[:elite_count]], axis=0)
            mean = elite_vectors.mean(axis=0).astype(np.float32)
            sigma = np.maximum(elite_vectors.std(axis=0).astype(np.float32), float(args.min_sigma))
            summary = {
                "iteration": iteration,
                "best_reward": best_reward,
                "iteration_best_reward": candidates[0][0],
                "mean_reward": float(np.mean([item[0] for item in candidates])),
                "sigma_mean": float(np.mean(sigma)),
                "target_finish_seconds": float(args.target_finish_seconds),
                "target_progress_speed": float(args.target_progress_speed),
                "best_planner_config": str(output_dir / "planner_config.json"),
                "best_weights": str(output_dir / "residual_weights.npz"),
            }
            save_json(output_dir / "training_summary.json", summary)

    print(
        json.dumps(
            {
                "best_reward": best_reward,
                "planner_config": str(output_dir / "planner_config.json"),
                "weights": str(output_dir / "residual_weights.npz"),
                "log": str(history_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
