import gym
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np
import matplotlib.pyplot as plt

# record = True
record = False
# save_video = True
save_video = False
env = VisionLandingAviary(
    aggregate_phy_steps=10,
    img_res= np.array([100, 100]),
    episode_len_sec=1.5,
    record=record,
    initial_xyzs=np.array([[0.3, 0.0, 0.6]], dtype=np.float64),
    initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    fov=80,
    difficulty=4,
)

obs = env.reset()
done = False
reward_sum = 0.0
env.render()
while not done:
    # action = env.action_space.sample()  # 임의의 액션 (학습 시에는 정책의 출력 사용)
    # action = np.array([0.5, 0.5, 0.2], dtype=np.float32)
    action = np.array([-0.5, 0.0, 0.1], dtype=np.float32)
    # action = np.array([1.0, 0.0, 0.3], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    reward_sum += reward
    # print(env.last_action)
    env.render()
    print("Reward:", reward)
    print("Done:", done)

print(f"\nTotal reward: {reward_sum}")
if record and save_video:
    env.convert_external_images_to_video()
    env.convert_onboard_images_to_video()
env.close()

print("Done")
