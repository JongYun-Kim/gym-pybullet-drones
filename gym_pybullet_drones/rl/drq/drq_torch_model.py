import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from ray.rllib.utils.typing import TensorType
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from gym.spaces import Box, Discrete
import numpy as np
from ray.rllib.utils.annotations import override


class DrQPolicyModel(TorchModelV2, nn.Module):
    """Policy model for DrQ with convolutional layers + dense layers"""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, shared_conv_layers=None):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Determine input channels from observation space and stacked state size
        # # MUST: '_disable_preprocessor_api' is set to True in the config to get the raw observation space from the env
        # # What if you have _disable_preprocessor_api=False: you must use obs_space.original_space instead of obs_space
        stacked_image_space = obs_space["images"]
        # input_channels = stacked_image_space.shape[0]  # Assumes format (C, H, W); channel first
        stacked_state_size = obs_space["drone_states"].shape[0]

        # [1] Conv layers
        if shared_conv_layers is not None:
            self.conv_layers = shared_conv_layers
        else:
            raise ValueError("shared_conv_layers must be provided for DrQPolicyModel.")
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros((1, *stacked_image_space.shape))
            conv_out = self.conv_layers(dummy_input)
            conv_out_size = conv_out.shape[1]
        conv_out_size = conv_out_size + stacked_state_size  # stacked drone states
        # [2] A linear layer with LayerNorm and tanh as per DrQ paper
        self.embed = nn.Sequential(
            nn.Linear(conv_out_size, 50),
            nn.LayerNorm(50),
            nn.Tanh()
        )
        # [3] Policy head (dense layers)
        self.dense = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        stacked_images = obs["images"]
        stacked_drone_states = obs["drone_states"]

        assert stacked_images.dtype == torch.float32, \
            f"Stacked images must be of type torch.float32, but got {stacked_images.dtype}."  # TODO: REMOVE THIS LATER

        normalized_images = stacked_images  # already normalized in the main model
        conv_out = self.conv_layers(normalized_images).detach()  # the actor optimizer does NOT update conv layers

        concat_conv_out = torch.cat((conv_out, stacked_drone_states), dim=1)
        embed_out = self.embed(concat_conv_out)

        policy_out = self.dense(embed_out)

        return policy_out, state

    def value_function(self):
        # Shouldn't be used in DrQ, but required by TorchModelV2
        raise NotImplementedError("DrQ does not use value_function() in the policy model.")


class DrQQModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, shared_conv_layers=None):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        if shared_conv_layers is not None:
            self.conv_layers = shared_conv_layers
        else:
            raise ValueError("shared_conv_layers must be provided for DrQQModel.")

        stacked_image_space = obs_space["images"]
        input_channels = stacked_image_space.shape[0]  # Assumes format (C, H, W); channel first
        stacked_state_size = obs_space["drone_states"].shape[0]

        with torch.no_grad():
            dummy_input = torch.zeros((1, *stacked_image_space.shape))
            conv_out = self.conv_layers(dummy_input)
            conv_out_size = conv_out.shape[1]
        conv_out_size = conv_out_size + stacked_state_size

        self.embed = nn.Sequential(
            nn.Linear(conv_out_size, 50),
            nn.LayerNorm(50),
            nn.Tanh()
        )

        self.dense = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        stacked_images = obs["images"]
        stacked_drone_states = obs["drone_states"]
        actions = input_dict["actions"]

        normalized_images = stacked_images  # already normalized in the main model
        conv_out = self.conv_layers(normalized_images)  # No detach; it is the critic optimizer that updates the layers
        concat_conv_out = torch.cat((conv_out, stacked_drone_states), dim=1)
        embed_out = self.embed(concat_conv_out)

        concat_conv_out = torch.cat((embed_out, actions), dim=1)
        q_out = self.dense(concat_conv_out)  # (B, 1)

        return q_out, state

    def value_function(self):
        # Shouldn't be used in DrQ, but required by the base class
        return NotImplementedError

class SharedConvLayers(nn.Module):
    """Shared convolutional layers for DrQ policy and Q models.

    This class is used to define the shared convolutional layers
    for both the policy and Q models in DrQ. It is designed to be
    reused across multiple models to reduce redundancy.
    """
    def __init__(self, input_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()  # Flattened !!
        )

    def forward(self, x):
        return self.conv_layers(x)

class DrQTorchModel(SACTorchModel):
    """SACTorchModel with optimized data augmentation for DrQ.

    Implements Data-regularized Q (DrQ) augmentation techniques:
    1. Random shift augmentation for pixel-based observations
    2. Q-value regularization through multiple augmentations

    The implementation is optimized using grid_sample for efficient
    image transformations and batched processing for multiple augmentations.
    """
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            policy_model_config=None,
            q_model_config=None,
            twin_q=False,
            initial_alpha=1.0,
            target_entropy=None,
            shift_pad=4,  # Padding for random shift
    ):
        self.shared_conv_layers = None

        # obs_shape = obs_space.shape

        self.shift_pad = shift_pad
        # Pre-compute fixed sampling grid parameters
        self.identity_grid = None
        self.padding_mode = 'border'  # Most similar to 'replicate' in the paper

        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
        )
        print(f"[DrQTorchModel.__init__()] DrQ model initialized with shift_pad={shift_pad}.")

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        input_dict["obs"]["images"] = input_dict["obs"]["images"].float() / 255.0 - 0.5
        return input_dict["obs"], state

    def random_shift(self, imgs):
        n, c, h, w = imgs.size()
        assert h == w
        padding = tuple([self.shift_pad] * 4)
        x = F.pad(imgs, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.shift_pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.shift_pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.shift_pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.shift_pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             # mode='bilinear',
                             # padding_mode=self.padding_mode,
                             padding_mode='zeros',
                             align_corners=False)

    def random_shift_error(self, imgs):
        """Apply random shift augmentation to images using grid_sample for efficiency.

        Args:
            imgs: Image tensor of shape [B, C, H, W]; dtype=torch.float32

        Returns:
            Augmented image tensor of same shape with random shifts
        """
        # Check if images have proper shape [B, C, H, W]
        if len(imgs.shape) < 4:
            return imgs

        batch_size, channels, height, width = imgs.shape
        pad = self.shift_pad

        # No augmentation case - return original
        if pad == 0:
            return imgs

        # Initialize identity grid if not already created or if batch size changed
        if (self.identity_grid is None or
                self.identity_grid.size(0) != batch_size or
                self.identity_grid.size(1) != height or
                self.identity_grid.size(2) != width):
                # self.identity_grid.size(2) != height or
                # self.identity_grid.size(3) != width):
            # Create normalized identity grid [-1, 1]
            self.identity_grid = torch.nn.functional.affine_grid(
                torch.eye(2, 3, device=imgs.device).unsqueeze(0).repeat(batch_size, 1, 1),
                [batch_size, channels, height, width],
                align_corners=False
            )

        # Random shifts in normalized coordinates [-1, 1]
        shift_x = 2.0 * pad / width * (torch.rand(batch_size, device=imgs.device) - 0.5)
        shift_y = 2.0 * pad / height * (torch.rand(batch_size, device=imgs.device) - 0.5)

        # Create the shifted grid
        grid = self.identity_grid.clone()
        grid[:, :, :, 0] = grid[:, :, :, 0] + shift_x.view(-1, 1, 1)
        grid[:, :, :, 1] = grid[:, :, :, 1] + shift_y.view(-1, 1, 1)

        # Sample from the input image using the shifted grid
        return F.grid_sample(
            imgs,
            grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=False
        )

    def augment_obs(self, obs):
        """Apply appropriate augmentation based on observation type.

        Args:
            obs: Observation (tensor, dict of tensors, or other format)

        Returns:
            Augmented observation of the same structure
        """
        # Handle different observation types
        if isinstance(obs, torch.Tensor):
            # If obs is an image tensor [B, C, H, W]
            if len(obs.shape) == 4:
                return self.random_shift(obs)
            # If obs is a single image [C, H, W]
            elif len(obs.shape) == 3:
                return self.random_shift(obs.unsqueeze(0)).squeeze(0)
        elif isinstance(obs, dict):
            # If obs is a dict, augment each image tensor
            augmented = {}
            for k, v in obs.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                    augmented[k] = self.augment_obs(v)
                else:
                    augmented[k] = v
            return augmented

        # If not a supported format, return as is
        print("[DrQTorchModel.augment_obs()] Warning: Unsupported obs format for augmentation. Returning original.")
        return obs

    def get_q_values(self, model_out, actions=None, num_augmentations=None):
        """Apply DrQ augmentation to Q-value computation.

        Implements equation (3) from the DrQ paper by applying M augmentations
        to the current observation and averaging Q-values.

        Args:
            model_out: Feature outputs from the model layers (SACTorchModel's forward -- usually input_dict["obs"])
            actions: Actions to return the Q-values for (shape: [B, A])
            num_augmentations: Number of augmentations to average over (M in the paper)

        Returns:
            Tuple of averaged Q-values ([BATCH_SIZE, 1]) and None
        """
        if num_augmentations is None or num_augmentations <= 1:
            # Use original implementation for single output
            return super().get_q_values(model_out, actions)

        # Slower implementation but straightforward; improve later
        q_values = []
        for _ in range(num_augmentations):
            aug_model_out = self.augment_obs(model_out)
            q, _ = self._get_q_value(aug_model_out, actions, self.q_net)
            q_values.append(q)

        return torch.stack(q_values).mean(dim=0), None  # (B, 1), None

    def get_twin_q_values(self, model_out, actions=None, num_augmentations=None):
        """Apply DrQ augmentation to twin Q-value computation.

        Same as get_q_values but for the twin Q-network.

        Args:
            model_out: Feature outputs from the model layers
            actions: Actions to return the Q-values for
            num_augmentations: Number of augmentations to average over (M in the paper)

        Returns:
            Tuple of averaged Q-values and None
        """
        if self.twin_q_net is None:
            return None, None

        if num_augmentations is None or num_augmentations <= 1:
            # Use original implementation for single output
            return super().get_twin_q_values(model_out, actions)

        twin_q_values = []
        for _ in range(num_augmentations):
            aug_model_out = self.augment_obs(model_out)
            q, _ = self._get_q_value(aug_model_out, actions, self.twin_q_net)
            twin_q_values.append(q)
        return torch.stack(twin_q_values).mean(dim=0), None

    def _get_q_value(self, model_out, actions, net):
        input_dict = {
            "obs": model_out,
            "actions": actions,
            "is_training": True
        }
        return net(input_dict, [], None)  # (B, 1), state==[]

    def get_action_model_outputs(
            self,
            model_out: Dict[str, TensorType],
            state_in: List[TensorType] = None,
            seq_lens: TensorType = None,
    ) -> (TensorType, List[TensorType]):
        """Returns distribution inputs and states given the output of
        policy.model().

        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.
        Args:
            model_out: Feature outputs from the model layers (result of doing `model(obs)`).
            state_in: None
            seq_lens: None
        Returns:
            TensorType: Distribution inputs for sampling actions.
        """
        if state_in is None:
            state_in = []

        assert isinstance(model_out, dict) and "images" in model_out and "drone_states" in model_out, \
            "DrQ only supports model_out as a dict with 'images' and 'drone_states' keys. " \
            "Please set `model_out` to a dict with 'images' and 'drone_states' keys in the model config."

        # Note: self.action_model expects 'input_dict' type as a subclass of ModelV2 in RLlib
        return self.action_model({"obs": model_out}, state_in, seq_lens)

    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        if self.shared_conv_layers is None:
            self.shared_conv_layers = SharedConvLayers(4)  # Initialize shared conv layers, which is dirty tho
        else:
            raise ValueError("shared_conv_layers must be provided for DrQPolicyModel.")
        model = DrQPolicyModel(
            obs_space=obs_space,
            action_space=self.action_space,
            num_outputs=num_outputs,
            model_config=policy_model_config,
            name=name,
            shared_conv_layers=self.shared_conv_layers,
        )
        assert name=="policy_model"  # TODO: remove this once tested
        # if policy_model_config is not None:
        #     raise f"Currently, DrQ only implements {model.__class__.__name__} as the policy model. " \
        #           f"Please set `policy_model_config` to None in the model config."
        return model

    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        model = DrQQModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=q_model_config,
            name=name,
            shared_conv_layers=self.shared_conv_layers,
        )
        assert name in ["q", "twin_q"]  # TODO: remove this once tested
        # if q_model_config is not None:
        #     raise f"Currently, DrQ only implements {model.__class__.__name__} as the Q model. " \
        #           f"Please set `q_model_config` to None in the model config."
        return model

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return self.action_model.variables()

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables() + (
            self.twin_q_net.variables() if self.twin_q_net else []
        )
