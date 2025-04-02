"""
Data-regularized Q (DrQ) Implementation Package

This package implements DrQ (Data-regularized Q), a method for enhancing
pixel-based model-free reinforcement learning using image augmentation and
Q-function regularization as described in the paper:

"Image Augmentation Is All You Need: Regularizing Deep Reinforcement
Learning from Pixels" by Denis Yarats et al.

Classes:
    DrQ: Main algorithm class, extends SAC
    DrQConfig: Configuration class for DrQ settings
    DrQTorchPolicy: PyTorch policy implementation with augmentation
    DrQTorchModel: Model class with image augmentation methods
"""

from gym_pybullet_drones.rl.drq.drq import DrQ, DrQConfig, example_config
from gym_pybullet_drones.rl.drq.drq_torch_policy import DrQTorchPolicy
from gym_pybullet_drones.rl.drq.drq_torch_model import DrQTorchModel

__all__ = [
    "DrQ",
    "DrQConfig",
    "DrQTorchPolicy",
    "DrQTorchModel",
    "example_config",
]