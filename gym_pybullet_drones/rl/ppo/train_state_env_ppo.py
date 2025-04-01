import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
#
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import StateLandingAviary
#
from gym_pybullet_drones.envs.BaseAviary import Physics, DroneModel
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np


class LogSuccessRateCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_info = episode.last_info_for()
        if last_info is not None:
            is_landed = last_info['is_landed']
            is_crashed = last_info['is_crashed']
            is_timeout = last_info['is_timeout']

            assert is_landed + is_crashed + is_timeout == 1, \
                f"One of is_landed, is_crashed, is_timeout must be True. " \
                f"Got: is_landed={is_landed}, is_crashed={is_crashed}, is_timeout={is_timeout}"

            success_rate = float(is_landed)
            crash_rate = float(is_crashed)
            timeout_rate = float(is_timeout)

            episode.custom_metrics["success_rate"] = success_rate
            episode.custom_metrics["crash_rate"] = crash_rate
            episode.custom_metrics["timeout_rate"] = timeout_rate


if __name__ == "__main__":
    # enable_ray_local_mode_for_debugging = True
    enable_ray_local_mode_for_debugging = False
    ray.init(local_mode=enable_ray_local_mode_for_debugging)

    enable_log_success_rate = True
    # enable_log_success_rate = False
    callback_ = LogSuccessRateCallback if enable_log_success_rate else None

    # Env
    env_config = {
        "drone_model": DroneModel.CF2X,
        "initial_xyzs": None,
        "initial_rpys": None,
        "physics": Physics.PYB,
        "freq": 300,
        "aggregate_phy_steps": 10,
        "gui": False,
        "record": False,
        "obs": ObservationType.BW,
        "act": ActionType.VEL,
        "channel_first": True,  # torch's nn.Conv2d(): channel_first=True
        "stack_size": 4,  # num of stacked frames
        "fov": 80.0,  # field of view of the drone's camera in degrees
        "img_res": np.array([84, 84]),  # num pixels of the square image
        "img_fps": 30,
        "episode_len_sec": 12.0,  # episode length in "seconds" (float!)
        "include_drone_state": True,
        "include_actions_in_obs": True,
        "difficulty": 4,
        "curriculum_configs": None,
        "reward_params": [140.0, -1.0, -0.01, 1.6, 0.375, 0.0],
    }
    env_name = "state_landing_env"
    register_env(env_name, lambda cfg: StateLandingAviary(**cfg))

    custom_reporter = CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "timesteps_total": "ts",
            "custom_metrics/success_rate_mean": "s_rate",
            "episode_reward_mean": "rwd_avg",
            "episode_reward_min": "rwd_min",
            "episode_reward_max": "rwd_max",
        }
    )

    tune.run(
        "PPO",
        name="state_land0401",
        progress_reporter=custom_reporter,
        local_dir="~/temps/debugging_only",
        stop={"training_iteration": 80},
        checkpoint_freq=0,
        # keep_checkpoints_num=0,
        checkpoint_at_end=False,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            "callbacks": callback_,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            },
            "num_gpus": 1,
            "num_workers": 11,
            "num_envs_per_worker": 2,
            "rollout_fragment_length": 450,
            "train_batch_size": 22*450,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 19,
            # "batch_mode": "complete_episodes",
            # "batch_mode": "truncate_episodes",
            #
            "evaluation_interval": 10,
            "evaluation_duration_unit": "episodes",
            "evaluation_duration": 50,
            "evaluation_num_workers": 11,
            "evaluation_config": {
                "explore": True,
            },
            #
            "lr": 1e-4,
            "vf_loss_coeff": 0.1, #tune.grid_search([0.01, 0.1, 0.25, 0.03, 0.07, 0.15]),
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.96,
            "kl_coeff": 0,
            "clip_param": 0.21,  # 0.3
            "vf_clip_param": 256,
            # "grad_clip": None,
            "kl_target": 0.01,
        },
    )

