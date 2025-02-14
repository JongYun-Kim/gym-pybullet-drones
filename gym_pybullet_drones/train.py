import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary


# TODOs
# - [ ] Add evaluation during training




