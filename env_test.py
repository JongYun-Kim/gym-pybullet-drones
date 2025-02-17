import gym
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np
import cv2
import matplotlib.pyplot as plt


env = VisionLandingAviary(
    drone_model=DroneModel.CF2X,
    gui=False,           # PyBullet GUI 창이 뜨므로 시각적으로 확인 가능
    obs=ObservationType.BW,
    act=ActionType.VEL,
    aggregate_phy_steps=10,
    img_res= np.array([500, 500]),
    episode_len_sec=2.0,
    stack_size=4,
    record=True,
    # record=False,
    initial_xyzs=np.array([[0.0, 0.0, 0.]], dtype=np.float64),
    fov=90,
)

obs = env.reset()
done = False
while not done:
    # action = env.action_space.sample()  # 임의의 액션 (학습 시에는 정책의 출력 사용)
    action = np.array([0.5, 0.5, 0.05, 0.2], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    # env.render()는 텍스트 출력이지만, GUI 창으로 시각적으로 확인 가능
    env.render()
    # observation은 stacked 이미지: (H, W, 3*stack_size)
    # 최근 프레임만 확인하려면 마지막 3채널을 분리
    # current_frame = obs[..., -3:]
    # cv2.imshow("Drone Camera", current_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    #
    # plt.imshow(current_frame)
    # plt.axis("off")
    # plt.show()

# cv2.destroyAllWindows()
env.close()

print("Done")
