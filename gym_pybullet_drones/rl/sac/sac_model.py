from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
# typing
from typing import List, Union, Dict, Optional, Tuple
# torch
import torch
import torch.nn as nn


class VisionLanderSACPolicyModel(TorchModelV2, nn.Module):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_space = obs_space.original_space

        obs_space_images = obs_space.original_space["images"]
        obs_space_drone_state = obs_space.original_space["drone_states"]

        input_channel_size = obs_space_images.shape[0]
        drone_states_size = obs_space_drone_state.shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel_size, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.embd = nn.Sequential(
            nn.Linear(32 * 35 * 35 + drone_states_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )
        self.dense = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        self._init_weights(self.encoder)
        self._init_weights(self.embd)

    def _init_weights(self, modules):
        # 모든 conv 및 fc layer의 가중치를 orthogonal initialization, bias는 0으로 초기화
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        stacked_images = input_dict["obs"]["images"]  # (B, 4, 84, 84)
        stacked_images = stacked_images / 255.0
        stacked_drone_states = input_dict["obs"]["drone_states"]  # (B, 28)

        enc_out = self.encoder(stacked_images)  # (B, 32 * 35 * 35)
        assert enc_out.dim() == 2, "Encoder output should be 2D tensor"
        assert enc_out.shape[0] == stacked_images.shape[0], f"Batch size mismatch, Got {enc_out.shape[0]}, expected {stacked_images.shape[0]}"
        assert enc_out.shape[1] == 32 * 35 * 35, f"Encoder output size mismatch, Got {enc_out.shape[1]}, expected {32 * 35 * 35}"

        embd_out = self.embd(torch.cat([enc_out, stacked_drone_states], dim=1))  # (B, 50)

        logits = self.dense(embd_out)  # (B, num_outputs)

        # Note: This if statement: for debugging purpose (i.e. breakpoint at either training or evaluation)
        if self.training:
            return logits, state
        else:
            return logits, state


class VisionLanderSACQModel(TorchModelV2, nn.Module):

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_space_images = obs_space[0]["images"]
        obs_space_drone_state = obs_space[0]["drone_states"]

        input_channel_size = obs_space_images.shape[0]  # =: stack_size
        drone_states_size = obs_space_drone_state.shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel_size, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.embd = nn.Sequential(
            nn.Linear(32 * 35 * 35 + drone_states_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )

        self.dense = nn.Sequential(
            nn.Linear(53, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        self._init_weights(self.encoder)
        self._init_weights(self.embd)

    def _init_weights(self, modules):
        # 모든 conv 및 fc layer의 가중치를 orthogonal initialization, bias는 0으로 초기화
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs"]

        # Extract observation and action (if present)
        if isinstance(obs, tuple) and len(obs) == 2:
            # Case: (obs_dict, action)
            obs_dict, action = obs
        else:
            # Case: just obs_dict (no action)
            raise ValueError("Expected obs to be a tuple of (obs_dict, action)")

        stacked_images = obs_dict["images"]  # (B, 4, 84, 84)
        stacked_images = stacked_images / 255.0
        stacked_drone_states = obs_dict["drone_states"]  # (B, 28)

        enc_out = self.encoder(stacked_images)
        assert enc_out.dim() == 2, "Encoder output should be 2D tensor"
        assert enc_out.shape[0] == stacked_images.shape[0], f"Batch size mismatch, Got {enc_out.shape[0]}, expected {stacked_images.shape[0]}"
        assert enc_out.shape[1] == 32 * 35 * 35, f"Encoder output size mismatch, Got {enc_out.shape[1]}, expected {32 * 35 * 35}"

        embd_out = self.embd(torch.cat([enc_out, stacked_drone_states], dim=1))

        assert action.dim() == 2, "Action should be 2D tensor"
        assert action.shape[0] == stacked_images.shape[0], f"Batch size mismatch, Got {action.shape[0]}, expected {stacked_images.shape[0]}"
        assert action.shape[1] == 3, f"Action size mismatch, Got {action.shape[1]}, expected 3"

        embd_out = torch.cat([embd_out, action], dim=1)
        q_out = self.dense(embd_out)

        # Note: This if statement: for debugging purpose (i.e. breakpoint at either training or evaluation)
        if self.training:
            return q_out, state
        else:
            return q_out, state
