import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary_2D, VisionLandingAviary
from drq import DrQ, DrQConfig

# from ray.rllib.algorithms.sac.sac import SAC


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
        "include_actions_in_obs": False,
        "reward_params": [140.0, -1.0, -0.01, 1.6, 0.375, 0.0],
    }
    env_name_2d = "vision_landing_aviary_env_2d"
    register_env(env_name_2d, lambda cfg: VisionLandingAviary_2D(**cfg))
    env_name = "vision_landing_aviary_env"
    register_env(env_name, lambda cfg: VisionLandingAviary(**cfg))

    custom_reporter = CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "timesteps_total": "ts",
            "custom_metrics/success_rate_mean": "s_rate",
            "episode_reward_mean": "rwd_avg",
            # "episode_reward_min": "rwd_min",
            # "episode_reward_max": "rwd_max",
            "info/learner/default_policy/learner_stats/alpha_loss": "a_loss",
            "info/learner/default_policy/learner_stats/alpha_value": "a_val",
        }
    )

    config = DrQConfig()

    # Configure algorithm parameters
    config.training(
        # DrQ-specific parameters
        num_q_target_augmentations=2,  # K in the paper
        num_q_augmentations=2,         # M in the paper
        shift_pad=4,                   # Padding for random shift
        #
        # SAC parameters
        gamma=0.99,
        # n_step=1,
        twin_q=True,
        initial_alpha=0.1,
        target_entropy="auto",
        tau=0.005,
        target_network_update_freq=2,
        train_batch_size=1024,
        # training_intensity=None,
        num_steps_sampled_before_learning_starts=1500,
        replay_buffer_config={"capacity": int(8e5)},
        store_buffer_in_checkpoints=False, # True,
        #
        # Optimizer settings
        # grad_clip=None,
        optimization_config={
            "actor_learning_rate": 8e-4,
            "critic_learning_rate": 8e-4,
            "entropy_learning_rate": 8e-4,
        }
    )

    config.reporting(min_sample_timesteps_per_iteration=3000)

    config.environment(
        # env=env_name_2d,
        env=env_name,
        env_config=env_config,
        # clip_actions=True,
        # normalize_actions=True,
    )

    config.resources(
        num_gpus=1,
        # num_cpus_per_worker=1,
        # num_gpus_per_worker=0,
    )

    config.rollouts(
        num_rollout_workers=32,
        num_envs_per_worker=1,
        rollout_fragment_length=1,
        batch_mode=tune.grid_search(["complete_episodes", "truncate_episodes"]),
        # no_done_at_end=tune.grid_search([True, False]),
    )

    config.callbacks(LogSuccessRateCallback)

    # config.evaluation(
    #     evaluation_interval=10,
    #     evaluation_duration_unit="episodes",
    #     evaluation_duration=50,
    #     evaluation_num_workers=8,
    #     evaluation_config={
    #         "explore": True,
    #         "horizon": None,
    #         "no_done_at_end": False,
    #     },
    # )

    config.framework("torch")

    # # Run training
    # result = tune.run(
    #     DrQ,
    #     config=config.to_dict(),
    #     stop={"timesteps_total": 1_000_000},
    #     checkpoint_freq=10,
    #     checkpoint_at_end=True,
    #     name="drq_cartpole",
    #     local_dir="./results",
    # )

    config.experimental(_disable_preprocessor_api=True)

    results = tune.run(
        DrQ,
        name="drq_test0404", #0301: reward balanced 0.18h 0.15v,
        local_dir="~/temps/debugging_only",
        progress_reporter=custom_reporter,
        metric="custom_metrics/success_rate_mean",
        mode="max",
        stop={"training_iteration": 256},
        checkpoint_freq=0,
        # keep_checkpoints_num=16,
        checkpoint_at_end=False,
        # checkpoint_score_attr="episode_reward_mean",
        checkpoint_score_attr="custom_metrics/success_rate_mean",
        config=config.to_dict(),
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
