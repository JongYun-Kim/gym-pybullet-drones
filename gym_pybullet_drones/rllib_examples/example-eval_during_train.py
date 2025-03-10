import numpy as np
import gym
from gym import spaces

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig


# 1. Custom Gym Environment
class MyCustomEnv(gym.Env):
    """
    A simple environment in which the agent tries to reach state >= 10
    before max_steps is hit. If state >= 10, set `info["is_success"] = True`.
    """
    def __init__(self, config=None):
        super(MyCustomEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # {0, 1, 2}

        self.target = 10
        self.max_steps = 20

        self.state = 0
        self.step_count = 0

    def reset(self):
        self.state = 0
        self.step_count = 0
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # action = 0 -> state -= 1
        # action = 1 -> no change
        # action = 2 -> state += 1
        self.state += (action - 1)

        reward = 0.0
        done = False
        is_success = False

        if self.state >= self.target:
            reward = 1.0
            done = True
            is_success = True
        elif self.step_count >= self.max_steps:
            done = True

        info = {"is_success": is_success}
        return np.array([self.state], dtype=np.float32), reward, done, info


# 2. Custom Callback to track success as a custom metric
class SuccessMetricCallback(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        last_info = episode.last_info_for()
        if last_info is not None:
            is_success = last_info.get("is_success", False)
            episode.custom_metrics["success"] = 1.0 if is_success else 0.0


if __name__ == "__main__":
    ray.init()
    # ray.init(local_mode=True)

    # 3. Create a PPOConfig and set up our evaluation parameters
    #    using `evaluation_duration` instead of `evaluation_num_episodes`.
    config = (
        PPOConfig()
        .framework(framework="torch")
        .environment(env=MyCustomEnv)
        .resources(num_gpus=1,)
        .rollouts(num_rollout_workers=8,
                  rollout_fragment_length=1000,
                  )
        .callbacks(SuccessMetricCallback)
        .training(model={"fcnet_hiddens": [128, 128]},
                  train_batch_size=8000,
                  sgd_minibatch_size=512,
                  num_sgd_iter=10,
                  )
        .evaluation(
            # Evaluate every N iterations (in this case: 1).
            evaluation_interval=1,
            # Instead of "evaluation_num_episodes=2", we do:
            evaluation_duration=100,              # -> run * episodes per evaluation
            evaluation_duration_unit="episodes",
            # Alternatively, we could do:
            # evaluation_duration=100,          # (100 timesteps)
            # evaluation_duration_unit="timesteps",
            #
            # Turn off exploration during evaluation:
            evaluation_config={"explore": True},
            # Create 1 separate worker for evaluation (instead of doing it in the main process):
            evaluation_num_workers=8
        )
        # to_dict() only needed if we pass to tune.run("PPO", config=...)
        .to_dict()
    )

    # 4. Use Ray Tune to run the experiment
    analysis = tune.run(
        "PPO",
        config=config,
        stop={"training_iteration": 20},  # Stop after 00 training iterations
        checkpoint_freq=1,              # Checkpoint every iteration (for illustration)
        local_dir="~/temps/examples_only",       # Results logging dir
        # verbose=1
    )

    # 5. Inspect results
    df = analysis.results_df
    print("=== Final Results ===")
    print(
        df[
            [
                "training_iteration",
                "episode_reward_mean",                 # Training reward mean
                "custom_metrics/success_mean",         # Training success rate
                "evaluation/episode_reward_mean",      # Evaluation reward mean
                "evaluation/custom_metrics/success_mean",
            ]
        ]
    )

