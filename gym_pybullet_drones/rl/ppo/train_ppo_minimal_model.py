import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary_2D
from gym_pybullet_drones.rl.ppo.vision_landing_minimal_model_ppo import VisionLanderMinimalPPO


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

    # [1] Ray init
    # ray.init(local_mode=True)
    ray.init(local_mode=False)

    # [3] Env
    # Register your custom environment
    env_config = {
        "episode_len_sec": 10.0,  # episode length in "seconds" (float!)
        "include_drone_state": True,
        "include_actions_in_obs": True,
        "reward_params": [140.0, -1.0, -0.01, 1.6, 0.375, 0.0],
    }
    env_name_2d = "vision_landing_aviary_env_2d"
    register_env(env_name_2d, lambda cfg: VisionLandingAviary_2D(**cfg))

    # Register your custom model
    model_name = "vision_lander_ppo"
    ModelCatalog.register_custom_model(model_name, VisionLanderMinimalPPO)

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

    results = tune.run(
        "PPO",
        name="minimal_2d_0313", #0301: reward balanced 0.18h 0.15v,
        local_dir="~/temps/debugging_only",
        progress_reporter=custom_reporter,
        metric="custom_metrics/success_rate_mean",
        mode="max",
        stop={"training_iteration": 128},
        checkpoint_freq=5,
        keep_checkpoints_num=16,
        checkpoint_at_end=False,
        # checkpoint_score_attr="episode_reward_mean",
        checkpoint_score_attr="custom_metrics/success_rate_mean",
        config={
            "env": env_name_2d,
            "env_config": env_config,
            "framework": "torch",
            "callbacks": LogSuccessRateCallback,
            "model": {"custom_model": model_name,},
            "num_gpus": 1,
            "num_workers": 11,
            "num_envs_per_worker": 2,
            "rollout_fragment_length": 450,
            "train_batch_size": 22*450,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 19,
            "evaluation_interval": 5,
            "evaluation_duration_unit": "episodes",
            "evaluation_duration": 50,
            "evaluation_num_workers": 11,
            "evaluation_config": {"explore": True, "horizon": None, "no_done_at_end": False},
            "lr": 1e-4,
            "vf_loss_coeff": tune.grid_search([0.01, 0.05, 0.1, 0.25, 0.5]),
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.96,
            "kl_coeff": 0,
            "clip_param": 0.21,  # 0.3
            "vf_clip_param": 256,
            "grad_clip": tune.grid_search([0.5, 1.0, 5.0, 10.0, 40.0]),
            "kl_target": 0.01,
        },
    )

    df = results.results_df
    print("Best checkpoint path: ", results.best_checkpoint)
    print("Best checkpoint reward: ", results.best_result["episode_reward_mean"])
    print("Best checkpoint success rate: ", results.best_result["custom_metrics"]["success_rate_mean"])
    print(
        df[
            [
                "training_iteration",
                "episode_reward_mean",                          # Training reward mean
                "custom_metrics/success_rate_mean",             # Training success rate
                # "evaluation/episode_reward_mean",               # Evaluation reward mean
                # "evaluation/custom_metrics/success_rate_mean",  # Evaluation success rate
            ]
        ]
    )
