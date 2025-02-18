import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.models.vision_landing_model import VisionLanderPPO, VisionLanderPPOConfig

from gym_pybullet_drones.envs.BaseAviary import Physics, DroneModel
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np


# TODOs
# - [ ] Add evaluation during training


if __name__ == "__main__":

    # do_debug = False
    do_debug = True

    if do_debug:
        ray.init(local_mode=True)


    # register your custom environment
    env_config = {
        "drone_model": DroneModel.CF2X,
        "initial_xyzs": None,
        "initial_rpys": None,
        "physics": Physics.PYB,
        "freq": 240,
        "aggregate_phy_steps": 10,
        "gui": False,
        "record": False,
        "obs": ObservationType.BW,
        "act": ActionType.VEL,
        "channel_first": True,  # nn.Conv2d() 사용시 channel_first=True
        "stack_size": 4,  # 이미지 프레임 stack 개수
        "fov": 60.0,  # drone 카메라 시야각 (degree)
        "img_res": np.array([84, 84]),  # original: np.array([64, 48])
        "img_fps": 24,
        "episode_len_sec": 5.0,  # 에피소드 길이 (초)
        "include_drone_state": True,
    }
    env_name = "vision_landing_aviary_env"
    register_env(env_name, lambda cfg: VisionLandingAviary(**cfg))

    # Set up custom model configuration
    my_config_instance = VisionLanderPPOConfig()
    my_config_instance.ru_debugging = True
    custom_model_config = {
        "config_instance": my_config_instance,
        "config_in_dict": my_config_instance.to_dict(),
    }

    # register your custom model
    model_name = "vision_lander_ppo"
    ModelCatalog.register_custom_model(model_name, VisionLanderPPO)

    # train
    tune.run(
        "PPO",
        name="test_run_deleteme_0217",
        # resume=True,
        # stop={"episode_reward_mean": -101},
        # stop={"training_iteration": 300},
        checkpoint_freq=1,
        keep_checkpoints_num=16,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        config={
            "env": env_name,
            "env_config": env_config,
            "framework": "torch",
            #
            # "callbacks": MyCallbacks,
            #
            "model": {
                "custom_model": model_name,
                "custom_model_config": custom_model_config,
                # "custom_action_dist": "det_cont_action_dist" if custom_model_config["use_deterministic_action_dist"] else None,
            },
            "num_gpus": 1,
            "num_workers": 2,
            "num_envs_per_worker": 1,
            "rollout_fragment_length": 120,
            "train_batch_size": 8*120,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 8,
            # "batch_mode": "complete_episodes",
            # "batch_mode": "truncate_episodes",
            "lr": 4e-5,
            # "lr_schedule": [[0, 2e-5],
            #                 [1e7, 1e-7],
            #                 ],
            # Must be fine-tuned when sharing vf-policy layers
            "vf_loss_coeff": 0.25,
            # In the...
            "use_critic": True,
            "use_gae": True,
            "gamma": 0.992,
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
            "clip_param": 0.22,  # 0.3
            "vf_clip_param": 256,
            # "grad_clip": None,
            "grad_clip": 0.5,
            "kl_target": 0.01,
        },
    )



