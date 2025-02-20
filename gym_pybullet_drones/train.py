import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.models.vision_landing_model import VisionLanderPPO, VisionLanderPPOConfig

from gym_pybullet_drones.envs.BaseAviary import Physics, DroneModel
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np

from utils.my_rllib_utils import create_multi_callbacks_from_classes

# TODOs
# - [ ] Add evaluation during training
# - [ ] Track log_std of the policy output


class LogGradAndWeightStatsCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Access the model from the policy
        model = algorithm.get_policy().model

        # Safeguard "custom_metrics" in result
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}

        modules_to_monitor = {
            "encoder": model.encoder,
            "embedding": model.embedding,
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
    def on_train_result(self, *, algorithm, result, **kwargs):
        print("\n@@@ on_train_result starts in CurriculumCallbacks @@@\n")

        # Useful metrics
        # training_iteration = result["training_iteration"]
        episode_total = result["episodes_total"]
        # episode_this_iter = result["episodes_this_iter"]
        # episode_reward_mean = result["episode_reward_mean"]
        timesteps_total = result["timesteps_total"]

        # Difficulty Logic
        # if timesteps_total < 1000:
        #     difficulty = 1
        # elif timesteps_total < 2000:
        #     difficulty = 2
        # elif timesteps_total < 3000:
        #     difficulty = 3
        # else:
        #     difficulty = 4
        if episode_total < 4000:
            difficulty = 1
        elif episode_total < 8000:
            difficulty = 2
        elif episode_total < 12000:
            difficulty = 3
        else:
            difficulty = 4
        print(f" @@@           current difficulty: {difficulty}                @@@\n")

        # Set difficulty
        algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env.set_difficulty(difficulty))
        )

        # Add more in on_train_result

    # End Callbacks


if __name__ == "__main__":

    # [1] Ray init
    do_debug = False
    # do_debug = True
    if do_debug:
        ray.init(local_mode=True)

    # [2] Callbacks
    # Determine whether to use
    #   1. Curriculum learning (CurriculumCallbacks) and
    #   2. Log grad and weight stats (LogGradAndWeightStatsCallbacks)
    do_curriculum_learning = False
    enable_log_grad_and_weight_stats = True
    callback_classes = []
    if do_curriculum_learning:
        callback_classes.append(CurriculumCallbacks)
    if enable_log_grad_and_weight_stats:
        callback_classes.append(LogGradAndWeightStatsCallbacks)
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
        "episode_len_sec": 30.0,  # episode length in "seconds" (float!)
        "include_drone_state": True,
        "difficulty": 1 if do_curriculum_learning else 4,
    }
    env_name = "vision_landing_aviary_env"
    register_env(env_name, lambda cfg: VisionLandingAviary(**cfg))

    # [4] Model
    # Set up custom model configuration
    my_config_instance = VisionLanderPPOConfig()
    my_config_instance.ru_debugging = True
    my_config_instance.use_layer_norm = True
    my_config_instance.use_anomaly_detection = True
    custom_model_config = {
        "config_instance": my_config_instance,
        "config_in_dict": my_config_instance.to_dict(),
    }
    # Register your custom model
    model_name = "vision_lander_ppo"
    ModelCatalog.register_custom_model(model_name, VisionLanderPPO)
    # ModelCatalog.register_custom_action_dist("squashed_gaussian", TorchSquashedGaussian)

    # [5] Train
    tune.run(
        "PPO",
        # name="hyprprm_tune-250220",
        name="delete_me_me",
        # resume=True,
        # stop={"episode_reward_mean": -101},
        # stop={"training_iteration": 300},
        checkpoint_freq=8,
        keep_checkpoints_num=16,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            #
            "callbacks": multi_callbacks,
            #
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
                # "custom_action_dist": "squashed_gaussian",
            },
            "num_gpus": 1,
            "num_workers": 22,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 900,
            "train_batch_size": 22*900,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 40,
            # "batch_mode": "complete_episodes",
            # "batch_mode": "truncate_episodes",
            "lr": 4e-5,
            "lr_schedule": [[0,     4e-5],
                            [2e6,   2e-5],
                            [2.5e6, 1.8e-5],
                            [3e6,   1.5e-5],
                            [3.5e6, 1.3e-5],
                            [4e6,   1e-5],
                            [4.5e6, 9e-6],
                            [5e6,   8e-6],
                            ],
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.10,
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.991,
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
            "vf_clip_param": 128,
            # "grad_clip": None,
            "grad_clip": 10.0,
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
    #             # "callbacks": CurriculumCallbacks if do_curriculum_learning else None,
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




