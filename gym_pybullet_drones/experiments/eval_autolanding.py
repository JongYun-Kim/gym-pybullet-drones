import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.models.vision_landing_model import VisionLanderPPO, VisionLanderPPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy


def build_env(env_dict):
    env_ = VisionLandingAviary(**env_dict)
    return env_


if __name__ == "__main__":

    model_name = "vision_lander_ppo"
    ModelCatalog.register_custom_model(model_name, VisionLanderPPO)

    ckp_num = 1128
    base_path = "/home/ray/gym-pybullet-drones/gym_pybullet_drones/experiments/ckps/currc/"
    trial_name = "e2229"
    policy_path_struc = "/policies/default_policy"
    ckp_path = base_path + trial_name + "/checkpoint_00" + str(ckp_num) + policy_path_struc
    try:
        policy = Policy.from_checkpoint(ckp_path)
    except FileNotFoundError:
        print("File not found. Please check the checkpoint number.")
        print(f"checkpoint_path: {ckp_path}")
        exit()
    policy.model.eval()
    policy.model.cfg.eval_mode = True


    env_config_dict = {
        # "record": False,
        "record": True,
        "difficulty": 4,
    }
    env = build_env(env_config_dict)

    done = False
    obs = env.reset()
    reward_sum = 0
    while not done:
        action = policy.compute_single_action(obs, explore=False)[0]
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        env.render()
    env.close()


