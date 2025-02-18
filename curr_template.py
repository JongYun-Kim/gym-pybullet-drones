# For env
import gym
import numpy as np
from gym.spaces import Box, Discrete
# For Callbacks in Ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# main
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig  # 또는 SACConfig


def reward_fn_sparse(state, action, next_state):
    # 난이도 1에 해당하는 보상 계산 로직
    # 예: 간단한 shaping, 단순 성공/실패
    return 0

def reward_fn_dense1(state, action, next_state):
    # 난이도 2에 해당
    return 1

def reward_fn_dense2(state, action, next_state):
    # 난이도 3에 해당
    return 2

# Wrapping LunarLander-v2 with custom reward functions
class MultiRewardLander(gym.Env):
    def __init__(self, env_config):
        super().__init__()
        # 내부적으로 LunarLander 환경 사용
        # self._base_env = gym.make("LunarLander-v2")
        self._base_env = gym.make("CartPole-v0")

        # 보상함수 set: env_config에 담을 수도 있고, 여기서 직접 관리해도 됨
        self.reward_fn_set = env_config.get("reward_fn_set", [])

        # 초기 난이도 (reward_fn_set의 인덱스)
        self.difficulty = env_config.get("initial_difficulty", 0)

        # 관측·행동 공간은 base_env 것 재활용
        self.observation_space = self._base_env.observation_space
        self.action_space = self._base_env.action_space

        self.state = None
        self.diam_s = 19.12  # diameter of the observation space; used in the reward functions

    def reset(self):
        obs = self._base_env.reset()
        self.state = obs
        return obs

    def step(self, action):
        next_obs, original_reward, done, info = self._base_env.step(action)

        # 현재 difficulty 단계에 맞는 보상 함수를 선택
        current_reward_fn = self.reward_fn_set[self.difficulty]
        custom_reward = current_reward_fn(self.state, action, next_obs)

        self.state = next_obs
        return next_obs, custom_reward, done, info

    def set_difficulty(self, new_diff):
        self.difficulty = min(new_diff, len(self.reward_fn_set) - 1)


class ToddlerCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # result 안에 이번 iteration의 정보가 들어있습니다.
        # 예: total_timesteps, episodes_total 등등
        total_timesteps = result["timesteps_total"]  # 전체 샘플(스텝) 수

        # 예시 스케줄:
        #  - 0 ~ 100k   -> difficulty = 0
        #  - 100k ~ 150k -> difficulty = 1
        #  - 150k ~ 300k -> difficulty = 2
        #  - 300k 이상 -> difficulty = 3
        if total_timesteps < 8_000:
            new_diff = 0
        elif total_timesteps < 16_000:
            new_diff = 1
        elif total_timesteps < 24_000:
            new_diff = 2
        else:
            new_diff = 3
        # 모든 워커의 Env에 대해 난이도 업데이트
        def set_diff_in_env(env):
            if hasattr(env, "set_difficulty"):
                env.set_difficulty(new_diff)
            else:
                raise ValueError("In CALLBACK: Env does not have 'set_difficulty' method")

        algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(set_diff_in_env)
        )

# main
def main():
    debug_me = True
    # debug_me = False

    ray.init(local_mode=debug_me)

    config = (
        PPOConfig()
        .environment(
            env=MultiRewardLander,
            env_config={
                "reward_fn_set": [reward_fn_sparse, reward_fn_dense1, reward_fn_dense2],
                "initial_difficulty": 0
            }
        )
        .callbacks(ToddlerCallback)
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        # TODO: Add other settings!!!!
    )

    tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"timesteps_total": 32000},
    )

if __name__ == "__main__":
    print("Start training...")
    main()

