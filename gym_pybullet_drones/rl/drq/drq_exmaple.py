import ray
from ray import tune
from drq import DrQ, DrQConfig

def train_drq_dm_control():
    """Train DrQ on DeepMind Control Suite environment."""
    ray.init()

    # Create DrQ configuration
    config = DrQConfig()

    # Configure algorithm parameters
    config.training(
        # DrQ-specific parameters
        num_q_target_augmentations=2,  # K in the paper
        num_q_augmentations=2,         # M in the paper
        shift_pad=4,                   # Padding for random shift

        # SAC parameters
        gamma=0.99,
        tau=0.005,
        target_entropy="auto",
        n_step=1,
        train_batch_size=256,
        initial_alpha=1.0,

        # Optimizer settings
        optimization={
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        }
    )

    # Configure environment
    config.environment(
        env="dm_control_suite_cartpole-swingup-v0",
        env_config={},
        clip_actions=True,
        normalize_actions=True,
    )

    # Configure resources
    config.resources(
        num_gpus=1,
        num_cpus_per_worker=1,
        num_gpus_per_worker=0,
    )

    # Configure rollout workers
    config.rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=1,
        batch_mode="truncate_episodes",
    )

    # Framework must be torch for DrQ
    config.framework("torch")

    # Run training
    result = tune.run(
        DrQ,
        config=config.to_dict(),
        stop={"timesteps_total": 1_000_000},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        name="drq_cartpole",
        local_dir="./results",
    )

    ray.shutdown()
    return result

def train_drq_atari():
    """Train DrQ on Atari environment."""
    ray.init()

    # Configure algorithm
    config = DrQConfig()

    # DrQ parameters
    config.training(
        num_q_target_augmentations=1,  # K=1 for Atari as per paper
        num_q_augmentations=1,         # M=1 for Atari as per paper
        shift_pad=4,

        # SAC parameters for discrete actions
        target_entropy="auto",
        train_batch_size=64,
        twin_q=True,

        # Buffer configuration
        replay_buffer_config={
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 100_000,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
        },

        # More steps before learning starts for exploration
        num_steps_sampled_before_learning_starts=20_000,
    )

    # Environment config for Atari
    config.environment(
        env="ALE/Pong-v5",
        env_config={
            "frameskip": 4,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        normalize_actions=False,  # Don't normalize discrete actions
    )

    # Preprocessing for Atari
    config.update({
        "model": {
            "grayscale": True,
            "zero_mean": False,
            "dim": 84,
            "framestack": True,
            "num_framestacks": 4,
        },
    })

    # Resources
    config.resources(
        num_gpus=1,
        num_cpus_per_worker=1,
    )

    # Workers
    config.rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=20,
    )

    # Framework must be torch for DrQ
    config.framework("torch")

    # Run training (limited to 100k environment steps as in DrQ paper)
    result = tune.run(
        DrQ,
        config=config.to_dict(),
        stop={"timesteps_total": 100_000},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        name="drq_pong",
        local_dir="./results",
    )

    ray.shutdown()
    return result

if __name__ == "__main__":
    # Choose which environment to train on
    train_drq_dm_control()
    # Uncomment to train on Atari
    # train_drq_atari()

