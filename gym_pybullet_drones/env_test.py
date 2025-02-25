import gym
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np
import cv2
import matplotlib.pyplot as plt


env = VisionLandingAviary(
    aggregate_phy_steps=10,
    img_res= np.array([100, 100]),
    episode_len_sec=2.0,
    record=True,
    # record=False,
    initial_xyzs=np.array([[0.0, 0.0, 0.5]], dtype=np.float64),
    initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    fov=80,
    difficulty=4,
)

obs = env.reset()
done = False
while not done:
    # action = env.action_space.sample()  # 임의의 액션 (학습 시에는 정책의 출력 사용)
    # action = np.array([0.5, 0.5, 0.2], dtype=np.float32)
    action = np.array([0.0, 0.0, -0.49], dtype=np.float32)
    # action = np.array([1.0, 0.0, 0.3], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    # print(env.last_action)
    # env.render()는 텍스트 출력이지만, GUI 창으로 시각적으로 확인 가능
    env.render()
    print("Reward:", reward)
    print("Done:", done)

env.close()

print("Done")
