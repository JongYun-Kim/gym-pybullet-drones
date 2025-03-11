import numpy as np
import gym
from gym import spaces

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig


# 1. Custom Gym Environment
class MyCustomEnv(gym.Env):
    """
    A simple environment in which the agent tries to reach state >= 10
    before max_steps is hit. If state >= 10, set `info["is_success"] = True`.
    """
    def __init__(self, config=None):
        super(MyCustomEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # {0, 1, 2}

        self.target = 10
        self.max_steps = 20

        self.state = 0
        self.step_count = 0

    def reset(self):
        self.state = 0
        self.step_count = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # action = 0 -> state -= 1
        # action = 1 -> no change
        # action = 2 -> state += 1
        self.state += (action - 1)

        reward = 0.0
        done = False
        is_success = False

        if self.state >= self.target:
            reward = 1.0
            done = True
            is_success = True
        elif self.step_count >= self.max_steps:
            done = True

        info = {"is_success": is_success}
        return np.array([self.state], dtype=np.float32), reward, done, info


# 2. Custom Callback to track success as a custom metric
class SuccessMetricCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_info = episode.last_info_for()
        if last_info is not None:
            is_success = last_info.get("is_success", False)
            episode.custom_metrics["success"] = 1.0 if is_success else 0.0


if __name__ == "__main__":
    ray.init()
    # ray.init(local_mode=True)

    # 3. Create a PPOConfig and set up our evaluation parameters
    #    using `evaluation_duration` instead of `evaluation_num_episodes`.
    config = (
        PPOConfig()
        .framework(framework="torch")
        .environment(env=MyCustomEnv)
        .resources(num_gpus=1,)
        .rollouts(num_rollout_workers=8,
                  rollout_fragment_length=1000,
                  )
        .callbacks(SuccessMetricCallback)
        .training(model={"fcnet_hiddens": [128, 128]},
                  train_batch_size=8000,
                  sgd_minibatch_size=512,
                  num_sgd_iter=10,
                  )
        .evaluation(
            # Evaluate every N iterations (in this case: 1).
            evaluation_interval=1,
            # Instead of "evaluation_num_episodes=2", we do:
            evaluation_duration=100,              # -> run * episodes per evaluation
            evaluation_duration_unit="episodes",
            # Alternatively, we could do:
            # evaluation_duration=100,          # (100 timesteps)
            # evaluation_duration_unit="timesteps",
            #
            # Turn off exploration during evaluation:
            evaluation_config={"explore": True},
            # Create 1 separate worker for evaluation (instead of doing it in the main process):
            evaluation_num_workers=8
        )
        # to_dict() only needed if we pass to tune.run("PPO", config=...)
        .to_dict()
    )

    # 4. Use Ray Tune to run the experiment
    analysis = tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 20},  # Stop after 00 training iterations
        checkpoint_freq=1,              # Checkpoint every iteration (for illustration)
        local_dir="~/temps/examples_only",       # Results logging dir
        # verbose=1
    )

    # 5. Inspect results
    df = analysis.results_df
    print("=== Final Results ===")
    print(
        df[
            [
                "training_iteration",
                "episode_reward_mean",                 # Training reward mean
                "custom_metrics/success_mean",         # Training success rate
                "evaluation/episode_reward_mean",      # Evaluation reward mean
                "evaluation/custom_metrics/success_mean",
            ]
        ]
    )


# >>> for c in df.columns:
# >>>     if "evaluation" in c:
# >>>         print(c)
#
# evaluation/episode_reward_max
# evaluation/episode_reward_min
# evaluation/episode_reward_mean
# evaluation/episode_len_mean
# evaluation/episodes_this_iter
# evaluation/num_faulty_episodes
# evaluation/num_agent_steps_sampled_this_iter
# evaluation/num_env_steps_sampled_this_iter
# evaluation/timesteps_this_iter
# evaluation/num_recreated_workers
# evaluation/num_healthy_workers
# config/evaluation_interval
# config/evaluation_duration
# config/evaluation_duration_unit
# config/evaluation_sample_timeout_s
# config/evaluation_parallel_to_training
# config/evaluation_num_workers
# config/always_attach_evaluation_results
# config/enable_async_evaluation
# config/in_evaluation
# config/evaluation_num_episodes
# config/input_evaluation
# evaluation/custom_metrics/success_mean
# evaluation/custom_metrics/success_min
# evaluation/custom_metrics/success_max
# evaluation/hist_stats/episode_reward
# evaluation/hist_stats/episode_lengths
# evaluation/sampler_perf/mean_raw_obs_processing_ms
# evaluation/sampler_perf/mean_inference_ms
# evaluation/sampler_perf/mean_action_processing_ms
# evaluation/sampler_perf/mean_env_wait_ms
# evaluation/sampler_perf/mean_env_render_ms
# config/evaluation_config/num_gpus
# config/evaluation_config/num_cpus_per_worker
# config/evaluation_config/num_gpus_per_worker
# config/evaluation_config/_fake_gpus
# config/evaluation_config/placement_strategy
# config/evaluation_config/eager_tracing
# config/evaluation_config/eager_max_retraces
# config/evaluation_config/env
# config/evaluation_config/observation_space
# config/evaluation_config/action_space
# config/evaluation_config/env_task_fn
# config/evaluation_config/render_env
# config/evaluation_config/clip_rewards
# config/evaluation_config/normalize_actions
# config/evaluation_config/clip_actions
# config/evaluation_config/disable_env_checking
# config/evaluation_config/num_workers
# config/evaluation_config/num_envs_per_worker
# config/evaluation_config/sample_collector
# config/evaluation_config/sample_async
# config/evaluation_config/enable_connectors
# config/evaluation_config/rollout_fragment_length
# config/evaluation_config/batch_mode
# config/evaluation_config/remote_worker_envs
# config/evaluation_config/remote_env_batch_wait_ms
# config/evaluation_config/validate_workers_after_construction
# config/evaluation_config/ignore_worker_failures
# config/evaluation_config/recreate_failed_workers
# config/evaluation_config/restart_failed_sub_environments
# config/evaluation_config/num_consecutive_worker_failures_tolerance
# config/evaluation_config/horizon
# config/evaluation_config/soft_horizon
# config/evaluation_config/no_done_at_end
# config/evaluation_config/preprocessor_pref
# config/evaluation_config/observation_filter
# config/evaluation_config/synchronize_filters
# config/evaluation_config/compress_observations
# config/evaluation_config/enable_tf1_exec_eagerly
# config/evaluation_config/sampler_perf_stats_ema_coef
# config/evaluation_config/gamma
# config/evaluation_config/lr
# config/evaluation_config/train_batch_size
# config/evaluation_config/explore
# config/evaluation_config/actions_in_input_normalized
# config/evaluation_config/postprocess_inputs
# config/evaluation_config/shuffle_buffer_size
# config/evaluation_config/output
# config/evaluation_config/output_compress_columns
# config/evaluation_config/output_max_file_size
# config/evaluation_config/evaluation_interval
# config/evaluation_config/evaluation_duration
# config/evaluation_config/evaluation_duration_unit
# config/evaluation_config/evaluation_sample_timeout_s
# config/evaluation_config/evaluation_parallel_to_training
# config/evaluation_config/ope_split_batch_by_episode
# config/evaluation_config/evaluation_num_workers
# config/evaluation_config/always_attach_evaluation_results
# config/evaluation_config/enable_async_evaluation
# config/evaluation_config/in_evaluation
# config/evaluation_config/sync_filters_on_rollout_workers_timeout_s
# config/evaluation_config/keep_per_episode_custom_metrics
# config/evaluation_config/metrics_episode_collection_timeout_s
# config/evaluation_config/metrics_num_episodes_for_smoothing
# config/evaluation_config/min_time_s_per_iteration
# config/evaluation_config/min_train_timesteps_per_iteration
# config/evaluation_config/min_sample_timesteps_per_iteration
# config/evaluation_config/export_native_model_files
# config/evaluation_config/logger_creator
# config/evaluation_config/logger_config
# config/evaluation_config/log_level
# config/evaluation_config/log_sys_usage
# config/evaluation_config/fake_sampler
# config/evaluation_config/seed
# config/evaluation_config/_tf_policy_handles_more_than_one_loss
# config/evaluation_config/_disable_preprocessor_api
# config/evaluation_config/_disable_action_flattening
# config/evaluation_config/_disable_execution_plan_api
# config/evaluation_config/simple_optimizer
# config/evaluation_config/monitor
# config/evaluation_config/evaluation_num_episodes
# config/evaluation_config/metrics_smoothing_episodes
# config/evaluation_config/timesteps_per_iteration
# config/evaluation_config/min_iter_time_s
# config/evaluation_config/collect_metrics_timeout
# config/evaluation_config/buffer_size
# config/evaluation_config/prioritized_replay
# config/evaluation_config/learning_starts
# config/evaluation_config/replay_batch_size
# config/evaluation_config/replay_sequence_length
# config/evaluation_config/replay_mode
# config/evaluation_config/prioritized_replay_alpha
# config/evaluation_config/prioritized_replay_beta
# config/evaluation_config/prioritized_replay_eps
# config/evaluation_config/min_time_s_per_reporting
# config/evaluation_config/min_train_timesteps_per_reporting
# config/evaluation_config/min_sample_timesteps_per_reporting
# config/evaluation_config/input_evaluation
# config/evaluation_config/lr_schedule
# config/evaluation_config/use_critic
# config/evaluation_config/use_gae
# config/evaluation_config/kl_coeff
# config/evaluation_config/sgd_minibatch_size
# config/evaluation_config/num_sgd_iter
# config/evaluation_config/shuffle_sequences
# config/evaluation_config/vf_loss_coeff
# config/evaluation_config/entropy_coeff
# config/evaluation_config/entropy_coeff_schedule
# config/evaluation_config/clip_param
# config/evaluation_config/vf_clip_param
# config/evaluation_config/grad_clip
# config/evaluation_config/kl_target
# config/evaluation_config/vf_share_layers
# config/evaluation_config/lambda
# config/evaluation_config/input
# config/evaluation_config/callbacks
# config/evaluation_config/create_env_on_driver
# config/evaluation_config/custom_eval_function
# config/evaluation_config/framework
# config/evaluation_config/num_cpus_for_driver
# config/evaluation_config/tf_session_args/intra_op_parallelism_threads
# config/evaluation_config/tf_session_args/inter_op_parallelism_threads
# config/evaluation_config/tf_session_args/log_device_placement
# config/evaluation_config/tf_session_args/allow_soft_placement
# config/evaluation_config/local_tf_session_args/intra_op_parallelism_threads
# config/evaluation_config/local_tf_session_args/inter_op_parallelism_threads
# config/evaluation_config/model/_use_default_native_models
# config/evaluation_config/model/_disable_preprocessor_api
# config/evaluation_config/model/_disable_action_flattening
# config/evaluation_config/model/fcnet_hiddens
# config/evaluation_config/model/fcnet_activation
# config/evaluation_config/model/conv_filters
# config/evaluation_config/model/conv_activation
# config/evaluation_config/model/post_fcnet_hiddens
# config/evaluation_config/model/post_fcnet_activation
# config/evaluation_config/model/free_log_std
# config/evaluation_config/model/no_final_linear
# config/evaluation_config/model/vf_share_layers
# config/evaluation_config/model/use_lstm
# config/evaluation_config/model/max_seq_len
# config/evaluation_config/model/lstm_cell_size
# config/evaluation_config/model/lstm_use_prev_action
# config/evaluation_config/model/lstm_use_prev_reward
# config/evaluation_config/model/_time_major
# config/evaluation_config/model/use_attention
# config/evaluation_config/model/attention_num_transformer_units
# config/evaluation_config/model/attention_dim
# config/evaluation_config/model/attention_num_heads
# config/evaluation_config/model/attention_head_dim
# config/evaluation_config/model/attention_memory_inference
# config/evaluation_config/model/attention_memory_training
# config/evaluation_config/model/attention_position_wise_mlp_dim
# config/evaluation_config/model/attention_init_gru_gate_bias
# config/evaluation_config/model/attention_use_n_prev_actions
# config/evaluation_config/model/attention_use_n_prev_rewards
# config/evaluation_config/model/framestack
# config/evaluation_config/model/dim
# config/evaluation_config/model/grayscale
# config/evaluation_config/model/zero_mean
# config/evaluation_config/model/custom_model
# config/evaluation_config/model/custom_action_dist
# config/evaluation_config/model/custom_preprocessor
# config/evaluation_config/model/lstm_use_prev_action_reward
# config/evaluation_config/exploration_config/type
# config/evaluation_config/evaluation_config/explore
# config/evaluation_config/multiagent/policy_map_capacity
# config/evaluation_config/multiagent/policy_map_cache
# config/evaluation_config/multiagent/policy_mapping_fn
# config/evaluation_config/multiagent/policies_to_train
# config/evaluation_config/multiagent/observation_fn
# config/evaluation_config/multiagent/count_steps_by
# config/evaluation_config/tf_session_args/gpu_options/allow_growth
# config/evaluation_config/tf_session_args/device_count/CPU
# config/evaluation_config/multiagent/policies/default_policy

