# drone_env.py
import gym
import numpy as np
from dm_env import specs, StepType
from dmc import ExtendedTimeStep

from gym_pybullet_drones.envs.single_agent_rl.VisionLandingAviary import VisionLandingAviary

class DroneGymWrapper:
    """
    Wrapper for custom drone environment to make it compatible with DrQ-v2.
    """
    def __init__(self, env_name, action_repeat=1, seed=None, landing_config=None):
        """
        Args:
            env_name: Name of the gym environment
            action_repeat: Number of times to repeat actions
            seed: Random seed
        """
        # self._env = gym.make(env_name)
        self._env = VisionLandingAviary(reward_params=landing_config['reward_params'],)
        if seed is not None:
            self._env.seed(seed)

        self._action_repeat = action_repeat

        # Process observation space
        obs_space = self._env.observation_space
        assert isinstance(obs_space, gym.spaces.Dict), "Expected Dict observation space"
        assert "images" in obs_space.spaces, "Expected 'images' in observation space"

        # Extract image shape - should be (4, 84, 84) for pre-stacked frames
        self._image_shape = obs_space.spaces["images"].shape

        # Create observation spec with just the image part
        self._obs_spec = specs.BoundedArray(
            shape=self._image_shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )

        # Convert gym action space to dm_env action spec
        action_space = self._env.action_space
        self._action_spec = specs.BoundedArray(
            shape=action_space.shape,
            dtype=np.float32,
            minimum=action_space.low,
            maximum=action_space.high,
            name='action'
        )

    def reset(self):
        """Reset the environment and return an initial time step."""
        obs_dict = self._env.reset()

        # Extract only the image part
        image = obs_dict["images"]

        return ExtendedTimeStep(
            observation=image,
            step_type=StepType.FIRST,
            action=np.zeros(self._action_spec.shape, dtype=self._action_spec.dtype),
            reward=0.0,
            discount=1.0
        )

    def step(self, action):
        """Apply action to the environment and return next time step."""
        reward = 0.0
        discount = 1.0

        for i in range(self._action_repeat):
            obs_dict, step_reward, done, info = self._env.step(action)
            reward += step_reward

            if done:
                # Episode ended
                step_type = StepType.LAST
                discount = 0.0
                break
        else:
            # Episode continuing
            step_type = StepType.MID

        # Extract only the image part
        image = obs_dict["images"]

        return ExtendedTimeStep(
            observation=image,
            step_type=step_type,
            action=action,
            reward=reward,
            discount=discount
        )

    def observation_spec(self):
        """Return the observation spec."""
        return self._obs_spec

    def action_spec(self):
        """Return the action spec."""
        return self._action_spec

    def close(self):
        """Close the environment."""
        self._env.close()

    # Support additional attributes accessed by DrQ-v2
    def __getattr__(self, name):
        return getattr(self._env, name)