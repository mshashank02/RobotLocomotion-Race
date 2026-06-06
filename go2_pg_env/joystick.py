"""Joystick locomotion task for the local Go2 environment.

This task is adapted from MuJoCo Playground's Go1 joystick task. The local
changes are intentionally small so that students can compare the official
baseline against a course-specific Go2 variant.

Observation summary
-------------------
state (actor input):
    [local_linvel(3), gyro(3), gravity(3),
     joint_pos_error(12), joint_vel(12),
     last_action(12), command(3)]  -> 48 dims

privileged_state (critic-only input during training):
    state + extra simulator-only signals -> 123 dims

Action summary
--------------
The policy outputs 12 joint offsets. The final motor target is:
    target_joint_pos = default_pose + action_scale * policy_action
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

from . import base as go2_base
from . import constants as consts


ACTOR_OBS_SIZE = 48
CRITIC_OBS_SIZE = 123
ACTION_SIZE = 12


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Task terms
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                command_tracking_error=-0.0,
                # Stability terms
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                dof_pos_limits=-1.0,
                pose=0.5,
                termination=-1.0,
                stand_still=-1.0,
                # Smoothness / efficiency terms
                torques=-0.0002,
                action_rate=-0.01,
                action_smoothness_1=-0.0,
                action_smoothness_2=-0.0,
                dof_acc=-0.0,
                energy=-0.001,
                # Foot-behavior terms
                feet_clearance=-2.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_impact_vel=-0.0,
                feet_air_time=0.1,
            ),
            tracking_sigma=0.25,
            tracking_sigma_yaw=0.25,
            max_foot_height=0.1,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            # Command sampling ranges for [vx, vy, yaw_rate]
            min=[-1.0, -0.4, -1.0],
            max=[1.0, 0.4, 1.0],
            # Probability that each command channel stays active
            b=[0.9, 0.25, 0.5],
            # Stage metadata is injected from configs/course_config.json.
            stage_name="stage_1",
            student_stage2_goal_min=[-1.0, -0.4, -1.0],
            student_stage2_goal_max=[1.0, 0.4, 1.0],
            student_stage2_goal_b=[0.9, 0.25, 0.5],
            resampling_time_seconds=10.0,
            curriculum=config_dict.create(
                enable=False,
                ema_alpha=0.05,
                success_threshold=0.8,
                error_threshold=[0.15, 0.15, 0.2],
                sampling_epsilon=0.05,
                min_bin_updates_before_expand=4,
                vx_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                vy_values=[-0.4, -0.2, 0.0, 0.2, 0.4],
                yaw_values=[-1.0, -0.5, 0.0, 0.5, 1.0],
                initial_vx_max_index=3,
                initial_vy_radius=0,
                initial_yaw_radius=0,
            ),
        ),
        impl="jax",
        naconmax=4 * 8192,
        njmax=40,
    )


def observation_layout() -> dict[str, list[tuple[str, int]]]:
    return {
        "state": [
            ("local_linvel", 3),
            ("gyro", 3),
            ("gravity", 3),
            ("joint_pos_error", 12),
            ("joint_vel", 12),
            ("last_action", 12),
            ("command", 3),
        ],
        "privileged_state_extra": [
            ("gyro_clean", 3),
            ("accelerometer", 3),
            ("gravity_clean", 3),
            ("local_linvel_clean", 3),
            ("global_angvel", 3),
            ("joint_pos_error_clean", 12),
            ("joint_vel_clean", 12),
            ("actuator_force", 12),
            ("last_contact", 4),
            ("feet_linvel", 12),
            ("feet_air_time", 4),
            ("external_force", 3),
            ("perturbation_active", 1),
        ],
    }


class Joystick(go2_base.Go2Env):
    """Track commanded planar velocity and yaw rate."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Joint limits skip the floating base.
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])
        self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])

        foot_linvel_sensor_adr = []
        for site in consts.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(sensor_adr, sensor_adr + sensor_dim)))
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._cmd_min = jp.array(self._config.command_config.min)
        self._cmd_max = jp.array(self._config.command_config.max)
        self._cmd_b = jp.array(self._config.command_config.b)
        self._command_stage_name = str(self._config.command_config.stage_name)
        self._student_stage2_goal_min = jp.array(self._config.command_config.student_stage2_goal_min)
        self._student_stage2_goal_max = jp.array(self._config.command_config.student_stage2_goal_max)
        self._student_stage2_goal_b = jp.array(self._config.command_config.student_stage2_goal_b)
        self._command_resampling_time = float(self._config.command_config.resampling_time_seconds)
        curriculum_cfg = self._config.command_config.curriculum
        self._command_curriculum_enabled = bool(curriculum_cfg.enable)
        self._command_curriculum_ema_alpha = float(curriculum_cfg.ema_alpha)
        self._command_curriculum_success_threshold = float(curriculum_cfg.success_threshold)
        self._command_curriculum_error_threshold = jp.array(curriculum_cfg.error_threshold)
        self._command_curriculum_sampling_epsilon = float(curriculum_cfg.sampling_epsilon)
        self._command_curriculum_min_bin_updates = float(curriculum_cfg.min_bin_updates_before_expand)
        self._curriculum_vx_values = jp.array(curriculum_cfg.vx_values)
        self._curriculum_vy_values = jp.array(curriculum_cfg.vy_values)
        self._curriculum_yaw_values = jp.array(curriculum_cfg.yaw_values)
        self._curriculum_vx_bins = len(curriculum_cfg.vx_values)
        self._curriculum_vy_bins = len(curriculum_cfg.vy_values)
        self._curriculum_yaw_bins = len(curriculum_cfg.yaw_values)
        self._curriculum_vy_center = int(np.argmin(np.abs(np.asarray(curriculum_cfg.vy_values, dtype=np.float32))))
        self._curriculum_yaw_center = int(np.argmin(np.abs(np.asarray(curriculum_cfg.yaw_values, dtype=np.float32))))
        self._curriculum_initial_limits = jp.array(
            [
                min(max(int(curriculum_cfg.initial_vx_max_index), 0), self._curriculum_vx_bins - 1),
                min(max(int(curriculum_cfg.initial_vy_radius), 0), self._curriculum_vy_center),
                min(max(int(curriculum_cfg.initial_yaw_radius), 0), self._curriculum_yaw_center),
            ],
            dtype=jp.int32,
        )
        self._curriculum_vx_edges = self._bin_edges(self._curriculum_vx_values, minimum=0.0, maximum=1.2)
        self._curriculum_vy_edges = self._bin_edges(self._curriculum_vy_values, minimum=-0.4, maximum=0.4)
        self._curriculum_yaw_edges = self._bin_edges(self._curriculum_yaw_values, minimum=-1.0, maximum=1.0)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize pose and base velocity a little at reset.
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self._config.pert_config.kick_wait_times[0],
            maxval=self._config.pert_config.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(jp.int32)

        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self._config.pert_config.kick_durations[0],
            maxval=self._config.pert_config.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(jp.int32)

        pert_mag = jax.random.uniform(
            key3,
            minval=self._config.pert_config.velocity_kick[0],
            maxval=self._config.pert_config.velocity_kick[1],
        )

        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * self._command_resampling_time
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(jp.int32)
        curriculum_limits = self._initial_curriculum_limits()
        command, command_bin = self._sample_command(
            key2,
            jp.zeros(3),
            curriculum_limits,
            jp.zeros((self._curriculum_vx_bins, self._curriculum_vy_bins, self._curriculum_yaw_bins)),
        )

        info = {
            "rng": rng,
            "command": command,
            "command_bin": command_bin,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_dof_vel": jp.zeros(self.mjx_model.nv - 6),
            "command_error_sum": jp.zeros(3),
            "command_steps": jp.zeros(()),
            "command_bin_success": jp.zeros(
                (self._curriculum_vx_bins, self._curriculum_vy_bins, self._curriculum_yaw_bins)
            ),
            "command_bin_count": jp.zeros(
                (self._curriculum_vx_bins, self._curriculum_vy_bins, self._curriculum_yaw_bins)
            ),
            "curriculum_limits": curriculum_limits,
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": steps_until_next_pert,
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
        }

        metrics = {f"reward/{name}": jp.zeros(()) for name in self._config.reward_config.scales.keys()}
        metrics["swing_peak"] = jp.zeros(())
        metrics.update(self._command_curriculum_metrics(info))

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if self._config.pert_config.enable:
            state = self._maybe_apply_perturbation(state)

        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        contact = jp.array(
            [data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0 for sensor_id in self._feet_floor_found_sensor]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt

        foot_positions = data.site_xpos[self._feet_site_id]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], foot_positions[..., -1])

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(data, action, state.info, state.metrics, done, first_contact, contact)
        rewards = {key: value * self._config.reward_config.scales[key] for key, value in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        state.info["command_error_sum"] += self._command_tracking_error(data, state.info["command"])
        state.info["command_steps"] += 1.0

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["last_dof_vel"] = data.qvel[6:]
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        resample_command = state.info["steps_until_next_cmd"] <= 0
        if self._command_curriculum_enabled and self._command_stage_name == "stage_2":
            self._update_command_curriculum(state.info, resample_command | done)
        new_command, new_command_bin = self._sample_command(
            key1,
            state.info["command"],
            state.info["curriculum_limits"],
            state.info["command_bin_success"],
        )
        state.info["command"] = jp.where(
            resample_command,
            new_command,
            state.info["command"],
        )
        state.info["command_bin"] = jp.where(resample_command, new_command_bin, state.info["command_bin"])
        state.info["steps_until_next_cmd"] = jp.where(
            done | resample_command,
            jp.round(jax.random.exponential(key2) * self._command_resampling_time / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )

        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        for key, value in rewards.items():
            state.metrics[f"reward/{key}"] = value
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        state.metrics.update(self._command_curriculum_metrics(state.info))

        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        # Terminate once the robot flips far enough that body-up is negative.
        return self.get_upvector(data)[-1] < 0.0

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> Dict[str, jax.Array]:
        # Noisy actor observations
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = gyro + (
            2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.gyro

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = gravity + (
            2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.gravity

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = joint_angles + (
            2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.joint_pos

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = joint_vel + (
            2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.joint_vel

        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = linvel + (
            2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1
        ) * self._config.noise_config.level * self._config.noise_config.scales.linvel

        state = jp.hstack(
            [
                noisy_linvel,
                noisy_gyro,
                noisy_gravity,
                noisy_joint_angles - self._default_pose,
                noisy_joint_vel,
                info["last_act"],
                info["command"],
            ]
        )

        # Critic-only privileged observations
        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        privileged_state = jp.hstack(
            [
                state,
                gyro,
                accelerometer,
                gravity,
                linvel,
                angvel,
                joint_angles - self._default_pose,
                joint_vel,
                data.actuator_force,
                info["last_contact"],
                feet_vel,
                info["feet_air_time"],
                data.xfrc_applied[self._torso_body_id, :3],
                info["steps_since_last_pert"] >= info["steps_until_next_pert"],
            ]
        )

        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(info["command"], self.get_local_linvel(data)),
            "tracking_ang_vel": self._reward_tracking_ang_vel(info["command"], self.get_gyro(data)),
            "command_tracking_error": self._cost_command_tracking_error(
                info["command"], self.get_local_linvel(data), self.get_gyro(data)
            ),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            "termination": self._cost_termination(done),
            "pose": self._reward_pose(data.qpos[7:]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "action_smoothness_1": self._cost_action_smoothness_1(action, info["last_act"]),
            "action_smoothness_2": self._cost_action_smoothness_2(action, info["last_act"], info["last_last_act"]),
            "dof_acc": self._cost_dof_acc(data.qvel[6:], info["last_dof_vel"]),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data),
            "feet_height": self._cost_feet_height(info["swing_peak"], first_contact, info),
            "feet_impact_vel": self._cost_feet_impact_vel(data, first_contact),
            "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        }

    # --- Task rewards ------------------------------------------------------

    def _reward_tracking_lin_vel(self, commands: jax.Array, local_vel: jax.Array) -> jax.Array:
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands: jax.Array, ang_vel: jax.Array) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma_yaw)

    def _cost_command_tracking_error(
        self, commands: jax.Array, local_vel: jax.Array, ang_vel: jax.Array
    ) -> jax.Array:
        lin_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        yaw_error = jp.square(commands[2] - ang_vel[2])
        return lin_error + 0.5 * yaw_error

    # --- Stability costs ---------------------------------------------------

    def _cost_lin_vel_z(self, global_linvel: jax.Array) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_stand_still(self, commands: jax.Array, qpos: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qpos - self._default_pose)) * (jp.linalg.norm(commands) < 0.01)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    # --- Smoothness and efficiency ----------------------------------------

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torques))

    def _cost_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        del last_last_act
        return jp.sum(jp.square(act - last_act))

    def _cost_action_smoothness_1(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - last_act))

    def _cost_action_smoothness_2(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.square(act - 2.0 * last_act + last_last_act))

    def _cost_dof_acc(self, qvel: jax.Array, last_qvel: jax.Array) -> jax.Array:
        return jp.sum(jp.square((qvel - last_qvel) / self.dt))

    # --- Foot behavior -----------------------------------------------------

    def _cost_feet_slip(self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        foot_weights = jp.array([1.0, 1.0, 1.8, 1.8])
        return jp.sum(vel_xy_norm_sq * contact * foot_weights) * (jp.linalg.norm(info["command"]) > 0.01)

    def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(self, swing_peak: jax.Array, first_contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (jp.linalg.norm(info["command"]) > 0.01)

    def _cost_feet_impact_vel(self, data: mjx.Data, first_contact: jax.Array) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        return jp.sum(jp.square(jp.minimum(feet_vel[..., 2], 0.0)) * first_contact)

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
        return jp.sum((air_time - 0.1) * first_contact) * (jp.linalg.norm(commands) > 0.01)

    # --- Perturbation and command sampling --------------------------------

    def _initial_curriculum_limits(self) -> jax.Array:
        if self._command_curriculum_enabled and self._command_stage_name == "stage_2":
            return self._curriculum_initial_limits
        return jp.array(
            [self._curriculum_vx_bins - 1, self._curriculum_vy_center, self._curriculum_yaw_center],
            dtype=jp.int32,
        )

    @staticmethod
    def _bin_edges(centers: jax.Array, *, minimum: float, maximum: float) -> jax.Array:
        mids = 0.5 * (centers[:-1] + centers[1:])
        return jp.concatenate([jp.array([minimum]), mids, jp.array([maximum])])

    def _command_tracking_error(self, data: mjx.Data, command: jax.Array) -> jax.Array:
        local_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        return jp.array(
            [
                jp.abs(command[0] - local_vel[0]),
                jp.abs(command[1] - local_vel[1]),
                jp.abs(command[2] - gyro[2]),
            ]
        )

    def _command_curriculum_metrics(self, info: dict[str, Any]) -> dict[str, jax.Array]:
        success = info["command_bin_success"]
        count = info["command_bin_count"]
        touched = count > 0.0
        mean_success = jp.where(jp.any(touched), jp.sum(success * touched) / jp.maximum(jp.sum(touched), 1.0), 0.0)
        limits = info["curriculum_limits"].astype(jp.float32)
        progress_vx = limits[0] / max(float(self._curriculum_vx_bins - 1), 1.0)
        progress_vy = limits[1] / max(float(self._curriculum_vy_center), 1.0)
        progress_yaw = limits[2] / max(float(self._curriculum_yaw_center), 1.0)
        return {
            "command_curriculum/progress_vx": progress_vx,
            "command_curriculum/progress_vy": progress_vy,
            "command_curriculum/progress_yaw": progress_yaw,
            "command_curriculum/active_bins": (limits[0] + 1.0) * (2.0 * limits[1] + 1.0) * (2.0 * limits[2] + 1.0),
            "command_curriculum/mean_bin_success": mean_success,
            "command_curriculum/current_bin_success": success[
                info["command_bin"][0], info["command_bin"][1], info["command_bin"][2]
            ],
        }

    def _update_command_curriculum(self, info: dict[str, Any], should_update: jax.Array) -> dict[str, Any]:
        command_steps = jp.maximum(info["command_steps"], 1.0)
        mean_error = info["command_error_sum"] / command_steps
        command_success = jp.all(mean_error < self._command_curriculum_error_threshold).astype(jp.float32)

        vx_idx, vy_idx, yaw_idx = info["command_bin"]
        old_success = info["command_bin_success"][vx_idx, vy_idx, yaw_idx]
        new_success = (
            (1.0 - self._command_curriculum_ema_alpha) * old_success
            + self._command_curriculum_ema_alpha * command_success
        )
        new_count = info["command_bin_count"][vx_idx, vy_idx, yaw_idx] + 1.0

        info["command_bin_success"] = info["command_bin_success"].at[vx_idx, vy_idx, yaw_idx].set(
            jp.where(should_update, new_success, old_success)
        )
        info["command_bin_count"] = info["command_bin_count"].at[vx_idx, vy_idx, yaw_idx].set(
            jp.where(should_update, new_count, info["command_bin_count"][vx_idx, vy_idx, yaw_idx])
        )

        successful_bin = (new_success >= self._command_curriculum_success_threshold) & (
            new_count >= self._command_curriculum_min_bin_updates
        )
        limits = info["curriculum_limits"]
        vx_frontier = vx_idx >= limits[0]
        vy_frontier = jp.abs(vy_idx - self._curriculum_vy_center) >= limits[1]
        yaw_frontier = jp.abs(yaw_idx - self._curriculum_yaw_center) >= limits[2]
        expand_vx = (successful_bin & vx_frontier).astype(jp.int32)
        expand_vy = (successful_bin & vy_frontier).astype(jp.int32)
        expand_yaw = (successful_bin & yaw_frontier).astype(jp.int32)
        expanded_limits = jp.array(
            [
                jp.minimum(limits[0] + expand_vx, self._curriculum_vx_bins - 1),
                jp.minimum(limits[1] + expand_vy, self._curriculum_vy_center),
                jp.minimum(limits[2] + expand_yaw, self._curriculum_yaw_center),
            ],
            dtype=jp.int32,
        )
        info["curriculum_limits"] = jp.where(should_update, expanded_limits, limits)
        info["command_error_sum"] = jp.where(should_update, jp.zeros(3), info["command_error_sum"])
        info["command_steps"] = jp.where(should_update, jp.zeros(()), info["command_steps"])
        return info

    def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
        def random_direction(rng: jax.Array) -> jax.Array:
            angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
            return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

        def apply_perturbation(inner_state: mjx_env.State) -> mjx_env.State:
            t = inner_state.info["pert_steps"] * self.dt
            envelope = 0.5 * jp.sin(jp.pi * t / inner_state.info["pert_duration_seconds"])
            force = (
                envelope
                * self._torso_mass
                * inner_state.info["pert_mag"]
                / inner_state.info["pert_duration_seconds"]
            )
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(force * inner_state.info["pert_dir"])
            data = inner_state.data.replace(xfrc_applied=xfrc_applied)
            inner_state = inner_state.replace(data=data)
            inner_state.info["steps_since_last_pert"] = jp.where(
                inner_state.info["pert_steps"] >= inner_state.info["pert_duration"],
                0,
                inner_state.info["steps_since_last_pert"],
            )
            inner_state.info["pert_steps"] += 1
            return inner_state

        def wait(inner_state: mjx_env.State) -> mjx_env.State:
            inner_state.info["rng"], rng = jax.random.split(inner_state.info["rng"])
            inner_state.info["steps_since_last_pert"] += 1
            xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
            data = inner_state.data.replace(xfrc_applied=xfrc_applied)
            inner_state.info["pert_steps"] = jp.where(
                inner_state.info["steps_since_last_pert"] >= inner_state.info["steps_until_next_pert"],
                0,
                inner_state.info["pert_steps"],
            )
            inner_state.info["pert_dir"] = jp.where(
                inner_state.info["steps_since_last_pert"] >= inner_state.info["steps_until_next_pert"],
                random_direction(rng),
                inner_state.info["pert_dir"],
            )
            return inner_state.replace(data=data)

        return jax.lax.cond(
            state.info["steps_since_last_pert"] >= state.info["steps_until_next_pert"],
            apply_perturbation,
            wait,
            state,
        )

    def _command_sampling_profile(
        self, current_command: jax.Array, curriculum_limits: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self._command_stage_name == "stage_2":
            return self._student_stage2_sampling_profile(current_command, curriculum_limits)
        return self._cmd_min, self._cmd_max, self._cmd_b

    def _student_stage2_sampling_profile(
        self, current_command: jax.Array, curriculum_limits: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Walk-these-ways-style adaptive stage_2 command curriculum.

        Continuous fallback for callers that do not use the discrete 3D grid.
        """
        del current_command
        limits = curriculum_limits.astype(jp.float32)
        progress = jp.array(
            [
                limits[0] / max(float(self._curriculum_vx_bins - 1), 1.0),
                limits[1] / max(float(self._curriculum_vy_center), 1.0),
                limits[2] / max(float(self._curriculum_yaw_center), 1.0),
            ]
        )
        cmd_min = self._cmd_min + progress * (self._student_stage2_goal_min - self._cmd_min)
        cmd_max = self._cmd_max + progress * (self._student_stage2_goal_max - self._cmd_max)
        cmd_keep_prob = self._cmd_b + progress * (self._student_stage2_goal_b - self._cmd_b)
        return cmd_min, cmd_max, cmd_keep_prob

    def _command_to_bins(self, command: jax.Array) -> jax.Array:
        span = jp.maximum(self._student_stage2_goal_max - self._student_stage2_goal_min, 1e-6)
        normalized = (command - self._student_stage2_goal_min) / span
        bins = jp.floor(
            jp.clip(normalized, 0.0, 1.0 - 1e-6)
            * jp.array([self._curriculum_vx_bins, self._curriculum_vy_bins, self._curriculum_yaw_bins])
        )
        return bins.astype(jp.int32)

    def _active_curriculum_mask(self, limits: jax.Array) -> jax.Array:
        vx_idx = jp.arange(self._curriculum_vx_bins)[:, None, None]
        vy_idx = jp.arange(self._curriculum_vy_bins)[None, :, None]
        yaw_idx = jp.arange(self._curriculum_yaw_bins)[None, None, :]
        return (
            (vx_idx <= limits[0])
            & (jp.abs(vy_idx - self._curriculum_vy_center) <= limits[1])
            & (jp.abs(yaw_idx - self._curriculum_yaw_center) <= limits[2])
        )

    def _sample_curriculum_cell(self, rng: jax.Array, limits: jax.Array, success: jax.Array) -> jax.Array:
        active = self._active_curriculum_mask(limits)
        weights = ((1.0 - success) + self._command_curriculum_sampling_epsilon) * active
        flat_weights = weights.reshape((-1,))
        flat_weights = flat_weights / jp.maximum(jp.sum(flat_weights), 1e-6)
        flat_idx = jax.random.choice(rng, flat_weights.shape[0], p=flat_weights)
        vx_idx = flat_idx // (self._curriculum_vy_bins * self._curriculum_yaw_bins)
        rem = flat_idx - vx_idx * self._curriculum_vy_bins * self._curriculum_yaw_bins
        vy_idx = rem // self._curriculum_yaw_bins
        yaw_idx = rem - vy_idx * self._curriculum_yaw_bins
        return jp.array([vx_idx, vy_idx, yaw_idx], dtype=jp.int32)

    def _sample_command_in_cell(self, rng: jax.Array, bin_idx: jax.Array) -> jax.Array:
        x_rng, y_rng, yaw_rng = jax.random.split(rng, 3)
        vx_idx, vy_idx, yaw_idx = bin_idx
        return jp.array(
            [
                jax.random.uniform(
                    x_rng,
                    (),
                    minval=self._curriculum_vx_edges[vx_idx],
                    maxval=self._curriculum_vx_edges[vx_idx + 1],
                ),
                jax.random.uniform(
                    y_rng,
                    (),
                    minval=self._curriculum_vy_edges[vy_idx],
                    maxval=self._curriculum_vy_edges[vy_idx + 1],
                ),
                jax.random.uniform(
                    yaw_rng,
                    (),
                    minval=self._curriculum_yaw_edges[yaw_idx],
                    maxval=self._curriculum_yaw_edges[yaw_idx + 1],
                ),
            ]
        )

    def _sample_command(
        self,
        rng: jax.Array,
        current_command: jax.Array,
        curriculum_limits: jax.Array,
        command_bin_success: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if self._command_curriculum_enabled and self._command_stage_name == "stage_2":
            cell_rng, command_rng = jax.random.split(rng)
            limits = curriculum_limits.astype(jp.int32)
            bin_idx = self._sample_curriculum_cell(cell_rng, limits, command_bin_success)
            command = self._sample_command_in_cell(command_rng, bin_idx)
            return command, bin_idx

        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        cmd_min, cmd_max, cmd_keep_prob = self._command_sampling_profile(current_command, curriculum_limits)
        candidate = jax.random.uniform(y_rng, shape=(3,), minval=cmd_min, maxval=cmd_max)
        active_mask = jax.random.bernoulli(z_rng, cmd_keep_prob, shape=(3,))
        blend_mask = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        command = current_command - blend_mask * (current_command - candidate * active_mask)
        return command, self._command_to_bins(command)

    def sample_command(self, rng: jax.Array, current_command: jax.Array) -> jax.Array:
        command, _ = self._sample_command(
            rng,
            current_command,
            self._initial_curriculum_limits(),
            jp.zeros((self._curriculum_vx_bins, self._curriculum_vy_bins, self._curriculum_yaw_bins)),
        )
        return command
