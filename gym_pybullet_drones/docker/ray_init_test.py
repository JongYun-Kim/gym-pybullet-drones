import torch
import ray
from ray import air, tune

gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")
gpu_count = torch.cuda.device_count() if gpu_available else 0

# For ray version 2.1.0 (it is using TURN.RUN !!! latest may not work.. it's le>
# Check if you have GPUs available on this container
ray.init()
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
        "num_gpus": gpu_count,
        "num_workers": 4,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 200,
        "train_batch_size": 800,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 8,
        "lr": 1e-4,
        "gamma": 0.99,
        "lambda": 0.96,
        "kl_coeff": 0,
        "kl_target": 0.01,
    },
)