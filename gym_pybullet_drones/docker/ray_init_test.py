import ray
from ray import air, tune

ray.init()

# For ray version 2.1.0 (it is using TURN.RUN !!! latest may not work.. it's le>
# Start with num_gpus=0
tune.run(
    "PPO",
    name="ray_setup_test",
    # stop={"episode_reward_mean": 200},
    stop={"training_iteration": 4},
    checkpoint_freq=2,
    keep_checkpoints_num=4,
    checkpoint_at_end=True,
    config={
        "env": "CartPole-v0",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 200,
        "train_batch_size": 800,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 8,
        "lr": 1e-4,
        "gamma": 0.992,
        "lambda": 0.96,
        "kl_coeff": 0,
        "kl_target": 0.01,
    },
)