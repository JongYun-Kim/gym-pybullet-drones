import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env import EnvContext
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env


# 1. 간단한 Custom Gym 환경
class SimpleSuccessEnv(gym.Env):
    def __init__(self, config: EnvContext):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)

        self.current_step = 0
        self.max_steps = 5
        self.cum_reward = None

    def reset(self):
        self.current_step = 0
        self.cum_reward = 0.0
        return np.array([0.0], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # 보상은 그냥 무작위 (예시)
        reward = 1.0 if np.random.rand() < 0.5 else 0.0
        self.cum_reward += reward

        # 에피소드 종료 조건
        done = (self.current_step >= self.max_steps)

        # done 시점에 특정 값 이상일 경우 성공 and info["success"]에 True 기록
        info = {}
        if done:
            if self.cum_reward > 3.5:
                info["success"] = True
            else:
                info["success"] = False

        obs = np.array([0.0], dtype=np.float32)
        return obs, reward, done, info


# 2. 콜백 클래스
class LogSuccessRateCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 에피소드가 끝날 때, info["success"]를 custom_metrics로 기록
        env_info = episode.last_info_for()  # (단일 에이전트 환경은 policy_id 불필요)
        if env_info and "success" in env_info:
            is_success = float(env_info["success"])  # True->1.0, False->0.0
            episode.custom_metrics["success_rate"] = is_success

# 환경 생성 함수 (register_env용)
def env_creator(env_config):
    return SimpleSuccessEnv(env_config)


# 3. PPO 학습 스크립트
if __name__ == "__main__":
    ray.init()

    register_env("SimpleSuccessEnv", env_creator)

    tune.run(
        "PPO",
        local_dir="~/temps/examples_only",  # 결과 저장 디렉토리 (기본값: ~/ray_results)
        stop={"training_iteration": 5},  # 5회 Iteration 후 종료
        config={
            "env": "SimpleSuccessEnv",     # 등록된 환경 이름
            "num_workers": 2,             # 여러 워커 사용
            "num_gpus": 1,
            "framework": "torch",
            "callbacks": LogSuccessRateCallbacks,  # 커스텀 콜백 사용
        },
    )

    # config = (
    #     PPOConfig()
    #     # .environment(env=SimpleSuccessEnv, env_config={})
    #     .environment(env="SimpleSuccessEnv", env_config={})
    #     .rollouts(num_rollout_workers=2)  # 여러 워커를 사용
    #     .framework("torch")               # 원하는 후레임워크 ㄱㄱ (torch;;)
    #     .callbacks(MyCallbacks)
    # )
    #
    # trainer = config.build()
    #
    # for i in range(5):
    #     result = trainer.train()
    #     print(
    #         f"Iteration {i} | "
    #         f"Reward: {result['episode_reward_mean']:.2f}, "
    #         f"SuccessRate: {result['custom_metrics'].get('success_rate_mean', 0.0):.2f}"
    #     )
