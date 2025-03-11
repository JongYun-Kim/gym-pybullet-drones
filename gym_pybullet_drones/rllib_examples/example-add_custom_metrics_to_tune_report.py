import gym
from gym.spaces import Discrete, Box
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune.progress_reporter import CLIReporter


# Dummy environment for demonstration
class DummyEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.current_step = 0
        self._max_episode_steps = 25

    def reset(self):
        self.current_step = 0
        return np.array([0.5], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        obs = np.array([0.5], dtype=np.float32)
        reward = 1.0

        if self.current_step >= self._max_episode_steps:
            done = True
            if np.random.rand() < 0.33:
                info = {"is_landed": 1, "is_crashed": 0, "is_timeout": 0}
            else:
                info = {"is_landed": 0, "is_crashed": 0, "is_timeout": 1}
        else:
            done = False
            info = {"is_landed": 0, "is_crashed": 0, "is_timeout": 0}

        return obs, reward, done, info


class LogSuccessRateCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_info = episode.last_info_for()
        if last_info is not None:
            is_landed = last_info['is_landed']
            is_crashed = last_info['is_crashed']
            is_timeout = last_info['is_timeout']

            assert is_landed + is_crashed + is_timeout == 1, (
                f"One of is_landed, is_crashed, is_timeout must be True. "
                f"Got: is_landed={is_landed}, is_crashed={is_crashed}, is_timeout={is_timeout}"
            )

            success_rate = float(is_landed)
            crash_rate = float(is_crashed)
            timeout_rate = float(is_timeout)

            episode.custom_metrics["success_rate"] = success_rate
            episode.custom_metrics["crash_rate"] = crash_rate
            episode.custom_metrics["timeout_rate"] = timeout_rate

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # custom_metrics에서 success_rate의 aggregate 값을 가져와 상위 키에 추가
        success_rate = result.get("custom_metrics", {}).get("success_rate_mean", -1)
        result["success_rate_mean"] = success_rate
        return result

if __name__ == "__main__":
    # Initialize Ray with local mode
    ray.init(local_mode=True)

    # PPOConfig
    config = (
        PPOConfig()
        .framework(framework="torch")
        .environment(env=DummyEnv)
        .resources(num_gpus=0,)
        .rollouts(
            num_rollout_workers=0,
            rollout_fragment_length=1000,
        )
        .callbacks(LogSuccessRateCallback)
        .training(
            model={"fcnet_hiddens": [32, 32]},
            train_batch_size=1000,
            sgd_minibatch_size=128,
            num_sgd_iter=8,
        )
        .to_dict()
    )

    # Set metrics to be reported to Tune's progress reporter (CLIReporter)
    reporter = CLIReporter(
        metric_columns=["training_iteration", "time_total_s", "success_rate_mean", "custom_metrics/success_rate_mean"],
    )

    # Run PPO with Tune (8 iterations)
    tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 8},
        progress_reporter=reporter
    )

    ray.shutdown()
