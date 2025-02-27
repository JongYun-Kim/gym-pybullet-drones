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

    ckp_num = "0150"  # MUST be a STRING !
    base_path = "/home/ray/temps/debugging_only/los_and_control_test_250226/"
    trial_name = "PPO_vision_landing_aviary_env_c363d_00000_0_2025-02-26_10-44-52"
    policy_path_struc = "/policies/default_policy"
    ckp_path = base_path + trial_name + "/checkpoint_00" + ckp_num + policy_path_struc
    try:
        policy = Policy.from_checkpoint(ckp_path)
    except FileNotFoundError:
        print("File not found. Please check the checkpoint number.")
        print(f"checkpoint_path: {ckp_path}")
        exit()
    policy.model.eval()
    policy.model.cfg.eval_mode = True

    # stochastic_action = False
    stochastic_action = True
    env_config_dict = {
        "record": False,
        # "record": True,
        "difficulty": 4,
    }
    env = build_env(env_config_dict)

    done = False
    obs = env.reset()
    env.render()
    reward_sum = 0

    # ----------------------------
    # 데이터 기록용 리스트 (시뮬레이션 동안)
    # ----------------------------
    time_steps = []         # 각 스텝 번호
    drone_positions = []    # 드론의 [x,y,z] 위치
    pad_positions = []      # 랜딩 패드의 [x,y,z] 위치
    altitudes = []          # 드론의 고도 (z 값)

    step_count = 0
    while not done:
        action_tuple = policy.compute_single_action(obs, explore=stochastic_action)
        obs, reward, done, info = env.step(action_tuple[0])
        reward_sum += reward

        # 각 스텝마다 현재 드론과 패드의 위치를 기록 (복사본 저장)
        time_steps.append(step_count)
        drone_positions.append(env.pos[0].copy())
        pad_positions.append(env.landing_pad_base_pos.copy())
        altitudes.append(env.pos[0][2])

        # env.render()
        print(f"Env step {int(env.step_counter//10)}: reward: {reward}")
        step_count += 1

    print(f"Episode reward: {reward_sum}")

    # 기록된 리스트를 numpy array로 변환
    drone_positions = np.array(drone_positions)  # shape: (N, 3)
    pad_positions = np.array(pad_positions)        # shape: (N, 3)
    altitudes = np.array(altitudes)
    time_steps = np.array(time_steps)
    final_step = time_steps[-1] if len(time_steps) > 0 else 0

    # -----------------------------------------
    # PLOT 1: Trajectories of Drone and Landing Pad in X-Y Plane (color-coded by time step)
    # - 드론: viridis, 패드: plasma 컬러맵 사용
    # - 컬러바는 플롯 바깥에 배치하며, 실제 도달한 time step 범위를 표시함.
    # -----------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scatter_drone = ax1.scatter(drone_positions[:, 0], drone_positions[:, 1],
                                c=time_steps, cmap="viridis", vmin=0, vmax=final_step,
                                label="Drone")
    scatter_pad = ax1.scatter(pad_positions[:, 0], pad_positions[:, 1],
                              c=time_steps, cmap="plasma", vmin=0, vmax=final_step,
                              label="Landing Pad")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("X-Y Trajectories of Drone and Landing Pad")
    ax1.legend()

    # 드론 컬러맵에 대한 컬러바 (플롯 오른쪽에 배치)
    cbar_drone = fig1.colorbar(scatter_drone, ax=ax1, pad=0.02)
    cbar_drone.set_label("Time Steps (Drone)")
    # 패드 컬러맵에 대한 컬러바 (플롯 오른쪽에 약간 떨어지게 배치)
    cbar_pad = fig1.colorbar(scatter_pad, ax=ax1, pad=0.15)
    cbar_pad.set_label("Time Steps (Pad)")

    # ---------------------------
    # PLOT 2: Drone Altitude over Time
    # - 가로: 시간, 세로: 고도
    # - 드론의 x-y 플롯과 동일한 viridis 컬러맵 사용하여 동일한 시간대를 참조
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter_alt = ax2.scatter(time_steps, altitudes,
                              c=time_steps, cmap="viridis", vmin=0, vmax=final_step)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Drone Altitude")
    ax2.set_title("Drone Altitude over Time")
    cbar_alt = fig2.colorbar(scatter_alt, ax=ax2, pad=0.02)
    cbar_alt.set_label("Time Steps")

    plt.show()
    plt.close()  # stop here to see them

    env.close()

