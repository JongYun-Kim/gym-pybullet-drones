import gym
import numpy as np
from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary
import pybullet as p


class MyEnv(VisionLandingAviary):
    def _get_random_drone_pose(self, initial_xyzs=None, initial_rpys=None):
        initial_xyzs = np.array([[0.0, 0.0, 4.0]])
        initial_rpys = np.array([[0.0, 0.0, 0.0]])

        return initial_xyzs, initial_rpys

    def _resetLandingPad(self):
        # 위치
        self.landing_pad_base_start_pos = np.array([0.0, 0.0, 0.0])
        self.landing_pad_base_pos = self.landing_pad_base_start_pos.tolist()
        # 헤딩각
        self.landing_pad_yaw = 0.0
        self.landing_pad_orientation = p.getQuaternionFromEuler([0, 0, self.landing_pad_yaw])
        # 속도
        self.landing_pad_speed = 0.495

    def _updateLandingPad(self):
        # dx, dy = 0.0, 0.0
        dt = self.TIMESTEP * self.AGGR_PHY_STEPS
        dx = self.landing_pad_speed * dt * np.cos(self.landing_pad_yaw)
        dy = self.landing_pad_speed * dt * np.sin(self.landing_pad_yaw)
        self.landing_pad_base_pos[0] += dx
        self.landing_pad_base_pos[1] += dy
        # z는 0.0으로 고정
        p.resetBasePositionAndOrientation(self.landing_pad_id,
                                          self.landing_pad_base_pos,
                                          self.landing_pad_orientation,
                                          physicsClientId=self.CLIENT)

record = True
# record = False
# save_video = True
save_video = False
env = MyEnv(
    img_res=np.array([100, 100]),
    episode_len_sec=10,
    record=record,
    # initial_xyzs=np.array([[1.1, 0.0, 1.0]], dtype=np.float64),
    # initial_rpys=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    reward_params=[140.0, -1.0, -0.01, 1.6, 0.375, 0.0],
)

obs = env.reset()
done = False
reward_sum = 0.0
env.render()
info = None
while not done:
    # action = env.action_space.sample()  # 임의의 액션 (학습 시에는 정책의 출력 사용)
    noise = np.random.randn(1) * 0.15
    action = np.array([0.2, noise, -0.5], dtype=np.float32)
    obs, reward, done, info = env.step(action)
    reward_sum += reward
    env.render()
    pad_pos = env._get_pad_center_position()
    print("Pad position:", pad_pos)
    print("Reward:", reward)
    print("Done:", done)

print(info) if info is not None else print("No info")
print(f"\nTotal reward: {reward_sum}")
if record and save_video:
    env.convert_external_images_to_video()
    env.convert_onboard_images_to_video()
env.close()

print("Done")
