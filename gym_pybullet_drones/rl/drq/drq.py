import ray
from ray.rllib.algorithms.sac.sac import SAC, SACConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import AlgorithmConfigDict, Type
from typing import Optional, Dict, Any

from gym_pybullet_drones.rl.drq.drq_torch_policy import DrQTorchPolicy

from ray.rllib.utils.deprecation import (
    # DEPRECATED_VALUE,
    # deprecation_warning,
    Deprecated,
)


class DrQConfig(SACConfig):
    """Configuration class for the DrQ algorithm.

    Data-regularized Q (DrQ) is an extension of Soft Actor-Critic (SAC)
    that uses image augmentation and Q-function regularization to improve
    sample efficiency when learning from pixel observations.

    Example:
        >>> config = DrQConfig().training(gamma=0.9, lr=0.01,
        ...     num_q_target_augmentations=2, num_q_augmentations=2)
        >>> algo = config.build(env="dm_control_suite_cartpole-swingup-v0")
        >>> algo.train()
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or DrQ)

        # DrQ-specific configuration
        self.num_q_target_augmentations = 2  # K in the paper
        self.num_q_augmentations = 2        # M in the paper
        self.shift_pad = 4                  # Padding for random shift

    @override(SACConfig)
    def training(
            self,
            *,
            num_q_target_augmentations: Optional[int] = None,
            num_q_augmentations: Optional[int] = None,
            shift_pad: Optional[int] = None,
            **kwargs,
    ) -> "DrQConfig":
        """Sets the training related configuration.

        Args:
            num_q_target_augmentations: Number of target augmentations (K in the paper).
                Used for regularizing target Q-values.
            num_q_augmentations: Number of Q-function augmentations (M in the paper).
                Used for regularizing current Q-values.
            shift_pad: Padding size for random shift augmentation in pixels.
                Higher values allow for larger shifts.
            **kwargs: Arguments to pass to SACConfig.training()

        Returns:
            This updated DrQConfig object.
        """
        # Pass kwargs onto super's training() method
        super().training(**kwargs)

        if num_q_target_augmentations is not None:
            self.num_q_target_augmentations = num_q_target_augmentations
        if num_q_augmentations is not None:
            self.num_q_augmentations = num_q_augmentations
        if shift_pad is not None:
            self.shift_pad = shift_pad

        return self


class DrQ(SAC):
    """Data-regularized Q (DrQ) algorithm class.

    DrQ enhances SAC with data augmentation and Q-function regularization
    to improve sample efficiency when learning from pixels.

    This implementation follows the paper:
    "Image Augmentation Is All You Need: Regularizing Deep Reinforcement
    Learning from Pixels" by Yarats et al.

    References:
        https://arxiv.org/abs/2004.13649
    """

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return DrQConfig().to_dict()

    @override(SAC)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return DrQTorchPolicy
        else:
            raise ValueError("DrQ only supports PyTorch framework")

    @override(SAC)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        """Validates the config and raises errors for invalid settings."""
        # Call super's validation method
        super().validate_config(config)

        # Ensure framework is torch
        if config["framework"] != "torch":
            raise ValueError("DrQ only supports PyTorch framework")

        # Validate DrQ-specific parameters
        if config.get("num_q_target_augmentations", 2) <= 0:
            raise ValueError("`num_q_target_augmentations` must be > 0")
        if config.get("num_q_augmentations", 2) <= 0:
            raise ValueError("`num_q_augmentations` must be > 0")
        if config.get("shift_pad", 4) < 0:
            raise ValueError("`shift_pad` must be >= 0")

# Example usage
def example_config():
    """Returns an example configuration for DrQ with optimal settings.

    Returns:
        A DrQConfig object with recommended settings for pixel-based control.
    """
    config = DrQConfig()
    config.training(
        # DrQ-specific settings
        num_q_target_augmentations=2,  # K parameter in paper
        num_q_augmentations=2,         # M parameter in paper
        shift_pad=4,                   # Random shift padding

        # SAC settings
        twin_q=True,
        target_entropy="auto",
        n_step=1,
        train_batch_size=512,

        # Optimizer settings
        optimization={
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        grad_clip=None,
    )

    return config

# Deprecated: Use ray.rllib.algorithms.sac.SACConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(DrQConfig().to_dict())

    @Deprecated(
        old="gym_pybullet_drones.rl.drq.drq::DEFAULT_CONFIG",
        new="gym_pybullet_drones.rl.drq.drq::SACConfig(...)",
        error=True,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()

