import ray
from ray import tune
from ray.tune.progress_reporter import CLIReporter
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary, VisionLandingAviary_2D
from gym_pybullet_drones.rl.ppo.vision_landing_model_ppo import VisionLanderPPO, VisionLanderPPOConfig

from gym_pybullet_drones.envs.BaseAviary import Physics, DroneModel
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np

from gym_pybullet_drones.utils.utils_my_rllib import create_multi_callbacks_from_classes
import copy

# TODOs
# - [x] Add evaluation during training
# - [ ] Track log_std of the policy output

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


class LogGradAndWeightStatsCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Access the model from the policy
        # model = algorithm.get_policy().model
        model: VisionLanderPPO = algorithm.get_policy().model

        # Safeguard "custom_metrics" in result
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}

        modules_to_monitor = {
            "actor_encoder": model.actor_encoder,
            "critic_encoder": model.critic_encoder,
            "actor_embedding": model.actor_embedding,
            "critic_embedding": model.critic_embedding,
        }
        for module_name, module in modules_to_monitor.items():
            # Ensure a nested dict exists for this module_name
            if module_name not in result["custom_metrics"]:
                result["custom_metrics"][module_name] = {}
            for layer_idx, layer in enumerate(module):
                for name, param in layer.named_parameters():
                    if param.requires_grad:
                        weight_norm = param.data.norm().item()
                        grad_norm = param.grad.data.norm().item() if param.grad is not None else float('nan')
                        max_weight = param.data.abs().max().item()
                        max_grad = param.grad.data.abs().max().item() if param.grad is not None else float('nan')

                        # Use a neat hierarchical naming pattern within each module
                        base_tag = f"L{layer_idx}/{name}"
                        result["custom_metrics"][module_name][f"{base_tag}_weight_norm"] = weight_norm
                        result["custom_metrics"][module_name][f"{base_tag}_grad_norm"] = grad_norm
                        result["custom_metrics"][module_name][f"{base_tag}_max_weight"] = max_weight
                        result["custom_metrics"][module_name][f"{base_tag}_max_grad"] = max_grad
                    else:
                        print(f"@@@ Parameter '{name}' in '{module_name}' layer {layer_idx} has no grad!! @@@")


class CurriculumCallbacks(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.difficulty_metric = None
        self.difficulty_plans = None
        self.difficulty_prev_iter = None

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        super().on_algorithm_init(algorithm=algorithm, **kwargs)
        self.difficulty_metric = algorithm.config.get("env_config",{}).get("curriculum_configs",{}).get("metric", None)
        self.difficulty_plans = algorithm.config.get("env_config",{}).get("curriculum_configs",{}).get("plans", None)
        if self.difficulty_metric is None or self.difficulty_plans is None:
            raise ValueError("CurriculumConfigs must be provided in env_config!")
        # if self.difficulty_metric not in ["timesteps_total", "episode_reward_mean", "training_iteration", "episodes_total", "episodes_this_iter"]:
        #     raise ValueError("CurriculumConfigs['metric'] must be one of ... them! See the if-clause above this line.")

    def on_train_result(self, *, algorithm, result, **kwargs):
        print("\n@@@ on_train_result starts in CurriculumCallbacks @@@\n")

        # Difficulty Logic
        metric_val = result.get(self.difficulty_metric, None)
        if metric_val is not None:
            difficulty = self._get_difficulty_from_plan(metric_val)
            if self.difficulty_prev_iter is not None and difficulty != self.difficulty_prev_iter:
                print(f" @@@   [{self.__class__.__name__}] changes difficulty from {self.difficulty_prev_iter} to {difficulty}   @@@\n")
            else:
                print(f" @@@   [{self.__class__.__name__}] keeps difficulty at {difficulty}   @@@\n")
            self.difficulty_prev_iter = difficulty
        else:
            raise ValueError(f"Metric '{self.difficulty_metric}' not found in result in 'on_train_result'!")

        # Set difficulty
        algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env.set_difficulty(difficulty))
        )

    def _get_difficulty_from_plan(self, metric_val):
        """
        예: difficulty_plan = [
            {"threshold":  800_000, "value": 1},
            {"threshold": 1_500_000, "value": 2},
            {"threshold": 2_500_000, "value": 3},
            {"threshold": float("inf"), "value": 4},
        ]
        """
        # Find the difficulty level
        for difficulty_minimum_one, plan in enumerate(self.difficulty_plans):
            if metric_val < plan:
                return difficulty_minimum_one + 1
        # If the metric value is larger than the last threshold
        # print if difficulty reaches the maximum if it's the first time
        if self.difficulty_prev_iter is None or self.difficulty_prev_iter != len(self.difficulty_plans) + 1:
            print(f" @@@   [{self.__class__.__name__}] reaches the max difficulty {len(self.difficulty_plans) + 1}   @@@\n")
        return len(self.difficulty_plans) + 1


def get_env_configs(default_env_config):
    reward_params_variants = [
        # [landing, crash,  viz,   scale, ratio,  time]
        [140.0,     -1.0,  -0.01,  0.6,   1.0,    0.0],
        # [  140.0,  -1.0,  -0.01,  1.6,   0.5,  -0.0],
        # [  180.0,  -1.0,  -0.01,  1.6,   0.5,  -0.1],
        # [  140.0,  -1.0,  -0.01,  0.5,   0.5,  -0.1],
        # [  140.0,  -40.0,  -0.1,  1.6,   0.5,  -0.05],
        # [  140.0,  -40.0,  -0.5,  1.6,   0.5,  -0.05],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [  000.0,   0.0,   0.0,   0.0,   0.00,   0.0],
        # [landing, crash, viz,   scale,  ratio,  time]
        # [  10.0,    0.0,   0.0,   0.0,   0.5,   0.0],
        # [  10.0,    0.0,   0.0,   0.0,   0.5,  -0.01],
        # [  10.0,   -1.0,   0.0,   0.0,   0.5,   0.0],  # 3
        # [  10.0,   -1.0,   0.0,   0.0,   0.5,  -0.01],
        # [  10.0,    0.0,  -0.01,  0.0,   0.5,   0.0],
        # [  10.0,    0.0,  -0.01,  0.0,   0.5,  -0.01],
        # [  10.0,   -1.0,  -0.01,  0.0,   0.5,   0.0],
        # [  10.0,   -1.0,  -0.01,  0.0,   0.5,  -0.01],
        # [  10.0,    0.0,   0.0,   0.01,  0.5,   0.0],  # 9  ;   10 + (450*0.01) = 14.5
        # [  10.0,    0.0,   0.0,   0.01,  0.5,  -0.01],  # 10
        # [  10.0,   -1.0,   0.0,   0.01,  0.5,   0.0],
        # [  10.0,   -1.0,   0.0,   0.01,  0.5,  -0.01],
        # [  10.0,    0.0,  -0.01,  0.01,  0.5,   0.0],
        # [  10.0,    0.0,  -0.01,  0.01,  0.5,  -0.01],
        # [  10.0,   -1.0,  -0.01,  0.01,  0.5,   0.0],
        # [  10.0,   -1.0,  -0.01,  0.01,  0.5,  -0.01],
        # [landing, crash, viz,   scale, ratio,  time]
        # [180.0,    -1.0,   -0.01, 1.6,   0.375,  0.0],
        # [180.0,    -1.0,   -0.01, 1.6,   0.375, -0.4],
        # [10.0,     -1.0,   -0.05,  0.05,  0.5,    0.0],
        # [10.0,     -10.0,  -0.05,  0.05,  0.5,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.05,  0.4,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.05,  0.5,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.05,  0.6,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.1,   0.4,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.1,   0.5,   -0.01], # 9
        # [10.0,     -1.0,   -0.05,  0.1,   0.6,   -0.01], # 10
        # [10.0,     -1.0,   -0.05,  0.01,  0.4,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.01,  0.5,   -0.01],
        # [10.0,     -1.0,   -0.05,  0.01,  0.6,   -0.01],
    ]
    # configs = [default_env_config]
    configs = []
    for params in reward_params_variants:
        new_config = copy.deepcopy(default_env_config)
        new_config["reward_params"] = params
        configs.append(new_config)
        #
        assert len(params) == 6
        assert params[0] > 0
        assert params[1] <= 0
        assert params[2] <= 0
        assert params[3] >= 0
        assert 0 <= params[4] <= 1
        assert params[5] <= 0
    return configs


if __name__ == "__main__":

    # [0] Run flags
    # # [0-1] ray init
    enable_ray_local_mode_for_debugging = True
    # enable_ray_local_mode_for_debugging = False
    # # [0-2] curriculum learning
    # enable_curriculum_learning = True
    enable_curriculum_learning = False
    for _ in range(4):
        print(f"\n!!! Curriculum learning is {'ENABLED' if enable_curriculum_learning else 'DISABLED'}")
        print(f"!!!   Make sure if you want to enable 'CURRICULUM LEARNING' or not.")
    # # [0-3] log grad and weight stats
    # enable_log_grad_and_weight_stats = True
    enable_log_grad_and_weight_stats = False
    # # [0-4] log success rate
    # enable_log_success_rate = True
    enable_log_success_rate = False

    # [1] Ray init
    ray.init(local_mode=enable_ray_local_mode_for_debugging)

    # [2] Curriculum Configurations (metrics and plans)
    if enable_curriculum_learning:
        curriculum_configs = {}
        # Choose a metric to track the curriculum
        #curriculum_configs["metric"] = "timesteps_total"
        # Plan the curriculum
        #plan_list = [
        #    800_000,
        #    1_500_000,
        #    2_500_000,
        #]
        curriculum_configs["metric"] = "training_iteration"
        plan_list = [10, 25, 40]
        num_difficulties = 4. # you can also automatically set this by len(plan_list) + 1
        assert len(plan_list) == num_difficulties - 1
        curriculum_configs["plans"] = plan_list
    else:
        curriculum_configs = None

    # [3] Callbacks
    # Determine whether to use
    #   1. Curriculum learning (CurriculumCallbacks) and
    #   2. Log grad and weight stats (LogGradAndWeightStatsCallbacks)
    callback_classes = []
    if enable_curriculum_learning:
        callback_classes.append(CurriculumCallbacks)
    if enable_log_grad_and_weight_stats:
        callback_classes.append(LogGradAndWeightStatsCallbacks)
    if enable_log_success_rate:
        callback_classes.append(LogSuccessRateCallback)
    multi_callbacks, callback_names = create_multi_callbacks_from_classes(callback_classes)

    # [3] Env
    # Register your custom environment
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
        "difficulty": 1 if enable_curriculum_learning else 4,
        "curriculum_configs": curriculum_configs,
        "reward_params": [140.0, -1.0, -0.01, 1.6, 0.375, 0.0],
    }
    env_configs_reward_test = get_env_configs(env_config)
    env_name = "vision_landing_aviary_env"
    register_env(env_name, lambda cfg: VisionLandingAviary(**cfg))
    env_name_2d = "vision_landing_aviary_env_2d"
    register_env(env_name_2d, lambda cfg: VisionLandingAviary_2D(**cfg))

    # [4] Model
    # Set up custom model configuration
    my_config_instance = VisionLanderPPOConfig()
    my_config_instance.ru_debugging = True
    my_config_instance.use_layer_norm = True
    my_config_instance.use_anomaly_detection = False
    my_config_instance.is_conv_shared = False
    custom_model_config = {
        "config_instance": my_config_instance,
        "config_in_dict": my_config_instance.to_dict(),
    }
    my_config_instance2 = VisionLanderPPOConfig()
    my_config_instance2.ru_debugging = True
    my_config_instance2.use_layer_norm = True
    my_config_instance2.use_anomaly_detection = False
    my_config_instance2.is_conv_shared = True
    custom_model_config2 = {
        "config_instance": my_config_instance2,
        "config_in_dict": my_config_instance2.to_dict(),
    }

    # Register your custom model
    model_name = "vision_lander_ppo"
    ModelCatalog.register_custom_model(model_name, VisionLanderPPO)

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

    # [5] Train
    tune.run(
        "PPO",
        # name="delete_me",  # just to see if any bugs in env and models
        name="image_test0401", #0301: reward balanced 0.18h 0.15v,
        progress_reporter=custom_reporter,
        local_dir="~/temps/debugging_only",
        # resume=True,
        # stop={"episode_reward_mean": -101},
        stop={"training_iteration": 80},
        checkpoint_freq=0,
        # keep_checkpoints_num=0,
        checkpoint_at_end=False,
        checkpoint_score_attr="episode_reward_mean",
        config={
            # "env": env_name,
            "env": env_name_2d,
            "env_config": env_config,
            # "env_config": env_configs_reward_test[0],
            # "env_config": tune.grid_search(env_configs_reward_test),
            "framework": "torch",
            #
            "callbacks": multi_callbacks,
            #
            "model": {
                "custom_model": model_name,
                # "custom_model_config": custom_model_config,
                "custom_model_config": custom_model_config2,
                # "custom_model_config": tune.grid_search([custom_model_config, custom_model_config2]),
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
            "evaluation_interval": 5,
            "evaluation_duration_unit": "episodes",
            "evaluation_duration": 50,
            "evaluation_num_workers": 11,
            "evaluation_config": {
                "explore": True,
                "horizon": None,
                "no_done_at_end": False,
            },
            #
            "lr": 1e-4,
            # "lr_schedule": [[0,     5e-5],
            #                 [2e5,   3e-5],
            #                 [2.5e5, 2.8e-5],
            #                 [3e5,   2.5e-5],
            #                 [3.5e5, 2.2e-5],
            #                 [4e5,   2e-5],
            #                 [4.5e5, 1.7e-5],
            #                 [1e6,   8e-6],
            #                 ],
            # Horizon settgins
            # "horizon": 360,  # YOU MUST CHECK IF THIS IS COMPATIBLE WITH YOUR ENV
            # "soft_horizon": False, # If True, the env will not reset, which we don't want; So, MUST be set to FALSE
            # "no_done_at_end": tune.grid_search([True, False]),
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": tune.grid_search([0.01, 0.1, 0.25, 0.03, 0.07, 0.15]),
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.99,
            "lambda": 0.96,
            "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
            # "entropy_coeff": tune.grid_search([0, 0.001, 0.0025, 0.01]),
            # "entropy_coeff_schedule": None,
            # "entropy_coeff_schedule": [[0, 0.003],
            #                            [5e4, 0.002],
            #                            [1e5, 0.001],
            #                            [2e5, 0.0005],
            #                            [5e5, 0.0002],
            #                            [1e6, 0.0001],
            #                            [2e6, 0],
            #                            ],
            "clip_param": 0.21,  # 0.3
            "vf_clip_param": 256,
            # "grad_clip": None,
            "grad_clip": 1.0,
            "kl_target": 0.01,
        },
    )
    #
    # Below is experimental; ignore it.
    #
    # tune.run_experiments({
    #     "nan_test0219": {
    #         "run": "PPO",
    #         "env": env_name,
    #         "env_config": env_config,
    #         "checkpoint_freq": 8,
    #         "keep_checkpoints_num": 16,
    #         "checkpoint_at_end": True,
    #         "checkpoint_score_attr": "episode_reward_mean",
    #         "config": {
    #             "framework": "torch",
    #             #
    #             # "callbacks": CurriculumCallbacks if enable_curriculum_learning else None,
    #             "callbacks": WatchEncoderGradNormCallbacks,
    #             #
    #             "model": {
    #                 "custom_model": model_name,
    #                 "custom_model_config": custom_model_config,
    #                 # "custom_action_dist": "squashed_gaussian",
    #             },
    #             "num_gpus": 1,
    #             "num_workers": 22,
    #             "num_envs_per_worker": 1,
    #             "rollout_fragment_length": 900,
    #             "train_batch_size": 22*900,
    #             "sgd_minibatch_size": 512,
    #             "num_sgd_iter": 40,
    #             # "batch_mode": "complete_episodes",
    #             # "batch_mode": "truncate_episodes",
    #             "lr": 5e-5,
    #             # "lr_schedule": [[0, 2e-5],
    #             #                 [1e7, 1e-7],
    #             #                 ],
    #             # Must be fine-tuned when sharing vf-policy layers
    #             "vf_loss_coeff": 0.10,
    #             # In the...
    #             "use_critic": True,
    #             "use_gae": True,
    #             "gamma": 0.991,
    #             "lambda": 0.96,
    #             "kl_coeff": 0,  # no PPO penalty term; we use PPO-clip anyway; if none zero, be careful Nan in tensors!
    #             # "entropy_coeff": tune.grid_search([0, 0.001, 0.0025, 0.01]),
    #             # "entropy_coeff_schedule": None,
    #             # "entropy_coeff_schedule": [[0, 0.003],
    #             #                            [5e4, 0.002],
    #             #                            [1e5, 0.001],
    #             #                            [2e5, 0.0005],
    #             #                            [5e5, 0.0002],
    #             #                            [1e6, 0.0001],
    #
    #             #                            [2e6, 0],
    #             "clip_param": 0.21, "vf_clip_param": 128,
    #             # "grad_clip": None,
    #             "grad_clip": 10.0,
    #             "kl_target": 0.01,
    #         },
    #     },
    #     "nan_test0219": {
    #         "run": "PPO",
    #         "env": env_name,
    #         "env_config": env_config,
    #         "checkpoint_freq": 8,
    #         "keep_checkpoints_num": 16,
    #         "checkpoint_at_end": True,




