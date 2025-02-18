import copy  # Remove this if not used (later)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
# typing
from typing import List, Union, Dict, Optional
from dataclasses import dataclass, field, asdict
# torch
import torch
import torch.nn as nn
import numpy as np
# custom modules
# from yours

import torch.distributions.distribution
@dataclass
class VisionLanderPPOConfig:
    ru_debugging: bool = False
    is_shared_net: bool= False
    # Encoder configs
    encoder_channels: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [2, 1, 1, 1])
    # policy, value configs
    embed_dim: int = 50  # encoder+drone_state -> 임베딩 후 차원
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [512, 512])
    value_hidden_sizes: List[int] = field(default_factory=lambda: [512, 512])
    # Add more if you need more model config

    def __post_init__(self):
        # (1) encoder 관련 validation
        # (1-1) encoder_channels, kernel_sizes, strides 타입 체크: List[int]
        if not all(isinstance(x, list) for x in [self.encoder_channels, self.kernel_sizes, self.strides]):
            raise ValueError(
                f"encoder_channels({type(self.encoder_channels)}), "
                f"kernel_sizes({type(self.kernel_sizes)}), strides({type(self.strides)}) "
                "타입이 모두 List[int]여야 합니다!")
        # (1-2) encoder_channels, kernel_sizes, strides 의 길이가 0보다 큰지
        if not all(len(x) > 0 for x in [self.encoder_channels, self.kernel_sizes, self.strides]):
            raise ValueError(
                f"encoder_channels({len(self.encoder_channels)}), "
                f"kernel_sizes({len(self.kernel_sizes)}), strides({len(self.strides)}) "
                "길이가 모두 0보다 커야 합니다!")
        # (1-3) encoder_channels, kernel_sizes, strides 의 길이가 동일한지
        if not (len(self.encoder_channels) == len(self.kernel_sizes) == len(self.strides)):
            raise ValueError(
                f"encoder_channels({len(self.encoder_channels)}), "
                f"kernel_sizes({len(self.kernel_sizes)}), strides({len(self.strides)}) "
                "길이가 모두 같아야 합니다!")
        # (1-4) 값이 모두 양수인지, 최대/최소 길이 제한 등
        if any(ch <= 0 for ch in self.encoder_channels):
            raise ValueError(f"encoder_channels는 모두 양수여야 합니다: {self.encoder_channels}")
        if any(k <= 0 for k in self.kernel_sizes):
            raise ValueError(f"kernel_sizes는 모두 양수여야 합니다: {self.kernel_sizes}")
        if any(s <= 0 for s in self.strides):
            raise ValueError(f"strides는 모두 양수여야 합니다: {self.strides}")

        # (2) policy, value 관련 validation
        # (2-1) embed_dim 타입 체크: int
        if not isinstance(self.embed_dim, int):
            raise ValueError(f"embed_dim({type(self.embed_dim)}) 타입이 int여야 합니다!")
        # (2-2) policy_hidden_sizes, value_hidden_sizes 타입 체크: List[int]
        if not all(isinstance(x, list) for x in [self.policy_hidden_sizes, self.value_hidden_sizes]):
            raise ValueError(
                f"policy_hidden_sizes({type(self.policy_hidden_sizes)}), "
                f"value_hidden_sizes({type(self.value_hidden_sizes)}) "
                "타입이 모두 List[int]여야 합니다!")
        # (2-3) policy_hidden_sizes, value_hidden_sizes 의 길이가 0보다 큰지
        if not all(len(x) > 0 for x in [self.policy_hidden_sizes, self.value_hidden_sizes]):
            raise ValueError(
                f"policy_hidden_sizes({len(self.policy_hidden_sizes)}), "
                f"value_hidden_sizes({len(self.value_hidden_sizes)}) "
                "길이가 모두 0보다 커야 합니다!")
        # (2-4) 값이 모두 양수인지, 최대/최소 길이 제한 등
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim은 양수여야 합니다: {self.embed_dim}")
        if any(ch <= 0 for ch in self.policy_hidden_sizes):
            raise ValueError(f"policy_hidden_sizes는 모두 양수여야 합니다: {self.policy_hidden_sizes}")
        if any(ch <= 0 for ch in self.value_hidden_sizes):
            raise ValueError(f"value_hidden_sizes는 모두 양수여야 합니다: {self.value_hidden_sizes}")

        # 필요하면 추가 validation...

    # dataclass에서 dict로 변환하기
    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        return asdict(self)


class VisionLanderPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # (0) Get model config
        # (0-1) Load model config
        if model_config is not None:
            self.cfg: VisionLanderPPOConfig = model_config["custom_model_config"]["config_instance"]
        else:
            raise ValueError("Model config is None! Please you MUST provide a model config in dict.")
        if self.cfg.ru_debugging:
            assert isinstance(self.cfg, VisionLanderPPOConfig), \
                f"model_config['custom_model_config'] is not VisionLanderPPOConfig. It is {type(self.cfg)}."
        # (0-2) Load obs_space, action_space
        obs_space_images = obs_space.original_space["images"]
        obs_space_drone_state = obs_space.original_space["drone_state"]

        input_channel_size = obs_space_images.shape[0]
        if self.cfg.ru_debugging:
            # action space size
            action_size = action_space.shape[0]
            assert action_size == 3, f"action_size is not 4! It is {action_size}."
            assert num_outputs == 2 * action_size, \
                f"num_outputs is not 2 * action_size! It is {num_outputs} and action_size is {action_size}."
            # obs_space 채널
            assert input_channel_size == 4, f"input_channel_size isn't 4! It is {input_channel_size}."

        # (0-3) Validate the model config (set default values if not provided)
        #   Here, the debug flag does not skip the validation as config error may be caused by human error.

        # (1) Encoder
        ## output size = (batch_size, 32, h_new, w_new)
        #### h_new = ((h_org - (k-1)) / s) - (k-1) - (k-1) - (k-1)  # stride > 1 only in the first layer
        #### e.g. in: (b, 4, 84, 84) -> out: (b, 32, 35, 35)
        #     - encoder_channels, kernel_sizes, strides 를 iterate 하여 layer 생성
        encoder_layers = []
        in_channels = input_channel_size
        for out_ch, k, s in zip(self.cfg.encoder_channels, self.cfg.kernel_sizes, self.cfg.strides):
            encoder_layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=k,
                stride=s
            ))
            encoder_layers.append(nn.ReLU())
            in_channels = out_ch  # match: prev-out -> next-in
        # Build the encoder
        self.encoder = nn.Sequential(*encoder_layers)
        # Get encoder output size using a sample data
        test_input = torch.zeros(1, obs_space_images.shape[0], obs_space_images.shape[1], obs_space_images.shape[2])
        conv_out = self.encoder(test_input)
        self._conv_out_flattened_size = int(np.prod(conv_out.shape[1:]))  # C * H * W
        # remove this assertion once stable
        if self.cfg.ru_debugging:
            assert self._conv_out_flattened_size == 32 * 35 * 35, (f"_conv_out_size is {self._conv_out_flattened_size}, "
                                                                   f"if you used non-default cfgs, please remove this line "
                                                                   f"or check the dim manually.")

        # (2) Embedding layer
        #     - encoder out(Flatten) + drone_state -> Linear -> BN -> Tanh
        #     - 최종적으로 embed_dim
        self.embedding = nn.Sequential(
            nn.Linear(self._conv_out_flattened_size + obs_space_drone_state.shape[0], self.cfg.embed_dim),
            nn.BatchNorm1d(self.cfg.embed_dim),
            nn.Tanh()
        )

        # (3) Policy network
        #     - structure: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
        #     - input: (batch_size, embed_dim)
        #     - output: (batch_size, 4)
        policy_layers = []
        in_size = self.cfg.embed_dim
        for out_size in self.cfg.policy_hidden_sizes:
            policy_layers.append(nn.Linear(in_size, out_size))
            policy_layers.append(nn.ReLU())
            in_size = out_size
        # 최종적으로 action_space.n(=4)차원으로 매핑
        # policy_layers.append(nn.Linear(in_size, self.action_space.n))
        policy_layers.append(nn.Linear(in_size, num_outputs))
        # Build the policy network
        # Output shape: (batch_size, num_outputs)
        self.policy = nn.Sequential(*policy_layers)

        # (4) Value network (critic)
        #     - structure: similar to policy network
        #     - input: (batch_size, embed_dim)
        #     - output: (batch_size, 1)
        value_layers = []
        in_size = self.cfg.embed_dim
        for out_size in self.cfg.value_hidden_sizes:
            value_layers.append(nn.Linear(in_size, out_size))
            value_layers.append(nn.ReLU())
            in_size = out_size
        value_layers.append(nn.Linear(in_size, 1))
        # Build the value network
        # Output shape: (batch_size, 1)
        self.critic = nn.Sequential(*value_layers)
        self._value_out = None  # store this for value_function() in forward()

    # def _validate_config(self) -> Dict[str, Union[str, int, float, bool]]:
    #     return NotImplementedError

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        batch_size = input_dict["obs"]["images"].shape[0]

        obs_dict = input_dict["obs"]
        stacked_images = obs_dict["images"]  # shape: (batch_size, 4, 84, 84)
        if self.cfg.ru_debugging:
            assert stacked_images.dtype == torch.float32, f"stacked_images.dtype: {stacked_images.dtype}"
        stacked_images /= 255.0  # Normalize the images
        drone_state = obs_dict["drone_state"]  # shape: (batch_size, 20)

        # (1) Encoder forward
        x = self.encoder(stacked_images)               # (batch_size, C, H, W)
        x = x.view(x.shape[0], -1)                     # (batch_size, C'*H'*W'==conv_out_dim)
        if self.cfg.ru_debugging:
            assert x.ndim == 2, f"Encoder output x: Not a 2D tensor!!\n  x.shape: {x.shape}"
        x = torch.cat([x, drone_state], dim=1)  # (batch_size, conv_out_dim + 21)

        # (2) Embedding
        x = self.embedding(x)                              # (batch_size, embed_dim)

        # (3) Policy (actor) forward
        logits = self.policy(x)                            # (batch_size, num_outputs)

        # (4) Value (critic) forward
        self._value_out = self.critic(x)             # (batch_size, 1)

        # (5) Check for NaN/Inf in the output
        if self.cfg.ru_debugging:
            if torch.isnan(logits).any():
                raise ValueError("logits에서 NaN 발생!")
            if torch.isinf(logits).any():
                raise ValueError("logits에서 Inf 발생!")

        return logits, state

    def value_function(self) -> TensorType:
        """
        forward()에서 계산된 self._value_out의 shape: (batch_size, 1).
        """
        assert self._value_out is not None, "Value head has not been computed yet!"
        return self._value_out.squeeze(-1)  # (batch_size,)
