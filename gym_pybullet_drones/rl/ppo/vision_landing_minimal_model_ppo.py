from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
# typing
from typing import List, Union, Dict, Optional, Tuple
# torch
import torch
import torch.nn as nn
import numpy as np
# from yours


class VisionLanderMinimalPPO(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_space_images = obs_space.original_space["images"]
        obs_space_drone_state = obs_space.original_space["drone_states"]

        input_channel_size = obs_space_images.shape[0]
        drone_states_size = obs_space_drone_state.shape[0]

        self.shared_encoder = nn.Sequential(
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

        self.actor_embd = nn.Sequential(
            nn.Linear(32 * 35 * 35 + drone_states_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )
        self.actor_dense = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs),
        )

        self.critic_embd = nn.Sequential(
            nn.Linear(32 * 35 * 35 + drone_states_size, 50),
            nn.LayerNorm(50),
            nn.Tanh(),
        )
        self.critic_dense = nn.Sequential(
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )
        self._value_out = None  # store this for value_function() in forward()

        self._init_weights(self.shared_encoder)
        self._init_weights(self.actor_embd)
        self._init_weights(self.critic_embd)

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

        stacked_images = input_dict["obs"]["images"]  # shape: (B, 4, 84, 84)
        stacked_drone_states = input_dict["obs"]["drone_states"]

        stacked_images = stacked_images / 255.0  # Not in-place operation

        actor_enc_out = self.shared_encoder(stacked_images).detach()  # only critic updates conv weights
        actor_input = torch.cat([actor_enc_out, stacked_drone_states], dim=1)
        actor_embd = self.actor_embd(actor_input)
        logits = self.actor_dense(actor_embd)  # (batch_size, num_outputs)

        critic_enc_out = self.shared_encoder(stacked_images)  # allow critic optimizer to update conv weights
        critic_input = torch.cat([critic_enc_out, stacked_drone_states], dim=1) # (b, combined_size)
        critic_embd = self.critic_embd(critic_input)  # (batch_size, embed_dim)
        self._value_out = self.critic_dense(critic_embd)  # (batch_size, 1)

        return logits, state

    def value_function(self) -> TensorType:
        """Squeezes the last dim of self._value_out from _forward_impl() and returns it."""
        assert self._value_out is not None, "Value head has not been computed yet!"
        return self._value_out.squeeze(-1)  # (batch_size,)
