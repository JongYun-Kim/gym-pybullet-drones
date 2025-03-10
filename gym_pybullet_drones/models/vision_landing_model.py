from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType
# typing
from typing import List, Union, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
# torch
import torch
import torch.nn as nn
import numpy as np
# for torch.autograd.detect_anomaly()
import contextlib
# from yours
from gym_pybullet_drones.models.modules.module_builders import build_encoder, build_embedding, build_mlp


@dataclass
class VisionLanderPPOConfig:
    # Debugging mode
    ru_debugging: bool = False
    use_anomaly_detection: bool = False
    # Evaluation mode
    eval_mode: bool = False
    # Network structure configs
    is_conv_shared: bool = False
    # Encoder configs
    encoder_channels: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [2, 1, 1, 1])
    # Embedding layer
    embed_dim: int = 50  # encoder+drone_state -> 임베딩 후 차원
    use_layer_norm: bool = True
    # policy, value configs
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [512, 512])
    value_hidden_sizes: List[int] = field(default_factory=lambda: [512, 512])
    # Add more if you need more model config

    def __post_init__(self):
        assert isinstance(self.ru_debugging, bool), f"ru_debugging({type(self.ru_debugging)}) 타입이 bool이어야 합니다!"
        assert isinstance(self.is_conv_shared, bool), f"is_shared_net({type(self.is_conv_shared)}) 타입이 bool이어야 합니다!"
        assert isinstance(self.use_anomaly_detection, bool), f"use_anomaly_detection({type(self.use_anomaly_detection)}) 타입이 bool이어야 합니다!"
        assert isinstance(self.eval_mode, bool), f"eval_mode({type(self.eval_mode)}) 타입이 bool이어야 합니다!"
        # (1) encoder 관련 validation
        # (1-1) encoder_channels, kernel_sizes, strides 타입 체크: List[int]
        if not all(isinstance(x, list) for x in [self.encoder_channels, self.kernel_sizes, self.strides]):
            raise ValueError(
                f"encoder_channels({type(self.encoder_channels)}), "
                f"kernel_sizes({type(self.kernel_sizes)}), strides({type(self.strides)}) "
                "All types of encoder_channels, kernel_sizes, strides must be List[int]!")
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
    """
    Network Architecture:
        - Actor: encoder -> embedding -> dense
        - Critic: encoder -> embedding -> dense; shares enc/embd w/ actor if is_shared_net=True
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # [1] Get model config
        self.cfg: VisionLanderPPOConfig = self._get_and_validate_model_config(model_config)

        # [2] Get obs/act space
        obs_space_images = obs_space.original_space["images"]
        obs_space_drone_state = obs_space.original_space["drone_states"]

        # [3] Get input/output sizes
        input_channel_size = obs_space_images.shape[0]  # assumes C,H,W; channel first
        drone_states_size = obs_space_drone_state.shape[0]  # assumes 1D;
        action_size = action_space.shape[0]
        if self.cfg.ru_debugging:
            self._validate_io_sizes(input_channel_size, drone_states_size, action_size, num_outputs)

        # [4] Build networks
        # [4-1] Actor encoder/embedding
        self.actor_encoder = build_encoder(
            input_channel_size,
            self.cfg.encoder_channels,
            self.cfg.kernel_sizes,
            self.cfg.strides
        )
        combined_size = self._get_combined_size(self.actor_encoder, input_channel_size,
                                                obs_space_images.shape[1], obs_space_images.shape[2], drone_states_size)
        self.actor_embedding = build_embedding(
            combined_size,
            self.cfg.embed_dim,
            self.cfg.use_layer_norm
        )

        # [4-2] Actor dense
        self.actor_dense = build_mlp(
            input_dim=self.cfg.embed_dim,
            hidden_sizes=self.cfg.policy_hidden_sizes,
            output_dim=num_outputs,
        )

        # [4-3] Critic encoder/embedding
        if self.cfg.is_conv_shared: # Shared network: reuse actor enc/embd; but set None-s for debugging purposes
            self.critic_encoder = None
        else:                      # Separate network: build encoder/embedding for critic
            self.critic_encoder = build_encoder(
                input_channel_size,
                self.cfg.encoder_channels,
                self.cfg.kernel_sizes,
                self.cfg.strides
            )

        # [4-4] Critic embedding
        self.critic_embedding = build_embedding(
            combined_size,
            self.cfg.embed_dim,
            self.cfg.use_layer_norm
        )

        # [4-5] Critic dense
        self.critic_dense = build_mlp(
            input_dim=self.cfg.embed_dim,
            hidden_sizes=self.cfg.value_hidden_sizes,
            output_dim=1,
        )

        self._value_out = None  # store this for value_function() in forward()

    def _get_and_validate_model_config(self, model_config):
        if model_config is not None:
            cfg: VisionLanderPPOConfig = model_config["custom_model_config"]["config_instance"]
        else:
            raise ValueError("Model config is None! Please you MUST provide a model config in dict.")
        if cfg.ru_debugging:
            assert isinstance(cfg, VisionLanderPPOConfig), \
                f"model_config['custom_model_config'] is not VisionLanderPPOConfig. It is {type(cfg)}."
        if cfg.ru_debugging:
            if cfg.is_conv_shared:
                print("[VisionLanderPPO] - Creating a SHARED network for actor and critic")
            else:
                print("[VisionLanderPPO] - Creating SEPARATE networks for actor and critic")
        return cfg

    def _validate_io_sizes(self, input_channel_size, drone_states_size, action_size, num_outputs):
        # Sorry 4 the magic numbers; update them as needed
        assert input_channel_size == 4, f"input_channel_size isn't 4! It is {input_channel_size}."
        assert drone_states_size in [28, 40], f"drone_state_size is not 28|40! It's {drone_states_size}."
        assert action_size == 3, f"action_size is not 3! It is {action_size}."
        assert num_outputs == 2 * action_size, \
            f"num_outputs is not 2 * action_size! It is {num_outputs} and action_size is {action_size}."

    def _get_combined_size(self, enc, c, h, w, drone_state_size):
        test_input = torch.zeros(1, c, h, w)
        enc_out = enc(test_input)
        conv_out_flattened_size = int(np.prod(enc_out.shape[1:]))
        if self.cfg.ru_debugging:
            # remove this assertion once stable
            assert conv_out_flattened_size == 32 * 35 * 35, (f"_conv_out_size is {self._conv_out_flattened_size}, "
                                                               f"if you used non-default cfgs, please remove this line "
                                                               f"or check the dim manually.")
        return conv_out_flattened_size + drone_state_size

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # Set context manager for anomaly detection
        if self.cfg.use_anomaly_detection:
            with torch.autograd.detect_anomaly():
                return self._forward_impl(input_dict, state, seq_lens)
        else:
            return self._forward_impl(input_dict, state, seq_lens)

    def _forward_impl(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs_dict = input_dict["obs"]
        stacked_images = obs_dict["images"]  # shape: (batch_size, stack_size==channel_size, H, W)
        stacked_drone_states = obs_dict["drone_states"]  # shape: (batch_size, drone_state_size)

        batch_size = stacked_images.shape[0]

        # 이미지 정규화 (복사 후 연산으로 변경)
        if self.cfg.ru_debugging:
            if self.cfg.eval_mode:
                assert stacked_images.dtype in [torch.uint8, torch.float32], f"stacked_images.dtype: {stacked_images.dtype}"
                # Type casting: not mandatory, but for readability (i.e. it's automatically done when normalized)
                stacked_images = stacked_images.float() if stacked_images.dtype == torch.uint8 else stacked_images.clone()
            else:
                assert stacked_images.dtype == torch.float32, f"stacked_images.dtype: {stacked_images.dtype}"
                stacked_images = stacked_images.clone()  # Create a copy to avoid in-place operation
        else:
            stacked_images = stacked_images.float() if stacked_images.dtype == torch.uint8 else stacked_images.clone()

        stacked_images = stacked_images / 255.0  # Not in-place operation

        # Actor: (images) ->[enc]-> enc_out+drone_state ->[embd]-> embd -> [dense]-> logits
        if self.cfg.is_conv_shared:
            actor_enc_out = self.actor_encoder(stacked_images).detach()  # only critic updates conv weights
        else:
            actor_enc_out = self.actor_encoder(stacked_images)
        actor_enc_flattened = actor_enc_out.reshape(batch_size, -1)
        actor_input = torch.cat([actor_enc_flattened, stacked_drone_states], dim=1)
        actor_embd = self.actor_embedding(actor_input)
        logits = self.actor_dense(actor_embd)  # (batch_size, num_outputs)

        # Critic: (images) ->[enc]-> enc_out+drone_state ->[embd]-> embd -> [dense]-> value
        if self.cfg.is_conv_shared:
            critic_enc_out = self.actor_encoder(stacked_images)  # allow critic optimizer to update conv weights
        else:
            critic_enc_out = self.critic_encoder(stacked_images)  # (batch_size, 32, 35, 35)

        critic_enc_flattened = critic_enc_out.reshape(batch_size, -1)  # (batch_size, conv_out_flattened_size)
        critic_input = torch.cat([critic_enc_flattened, stacked_drone_states], dim=1) # (b, combined_size)
        critic_embd = self.critic_embedding(critic_input)  # (batch_size, embed_dim)

        self._value_out = self.critic_dense(critic_embd)  # (batch_size, 1)

        # NaN/Inf 체크
        if self.cfg.ru_debugging and batch_size == 512:
            if torch.isnan(logits).any():
                print(f"@@@@@@@@@@@@@ stacked_images (nan): {torch.isnan(stacked_images).sum()}")
                print(f"@@@@@@@@@@@@@ drone_state (nan): {torch.isnan(stacked_drone_states).sum()}")
                print(f"@@@@@@@@@@@@@ actor_enc_out (nan): {torch.isnan(actor_enc_out).sum()}")
                print(f"@@@@@@@@@@@@@ actor_enc_flattened (nan): {torch.isnan(actor_enc_flattened).sum()}")
                print(f"@@@@@@@@@@@@@ actor_embd (nan): {torch.isnan(actor_embd).sum()}")

                if not self.cfg.is_conv_shared:
                    print(f"@@@@@@@@@@@@@ critic_enc_out (nan): {torch.isnan(critic_enc_out).sum()}")
                    print(f"@@@@@@@@@@@@@ critic_enc_flattened (nan): {torch.isnan(critic_enc_flattened).sum()}")
                    print(f"@@@@@@@@@@@@@ critic_embd (nan): {torch.isnan(critic_embd).sum()}")

                print(f"@@@@@@@@@@@@@ logits (nan): {torch.isnan(logits).sum(axis=0)}")
                print(f"@@@@@@@@@@@@@ self._value_out (nan): {torch.isnan(self._value_out).sum()}")
                print("logits에서 NaN 발생!")
            if torch.isinf(logits).any():
                raise ValueError("logits에서 Inf 발생!")

        return logits, state

    def value_function(self) -> TensorType:
        """Squeezes the last dim of self._value_out from _forward_impl() and returns it."""
        assert self._value_out is not None, "Value head has not been computed yet!"
        return self._value_out.squeeze(-1)  # (batch_size,)


class VisionLanderPPOLegacy(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # (0) Get model config
        if model_config is not None:
            self.cfg: VisionLanderPPOConfig = model_config["custom_model_config"]["config_instance"]
        else:
            raise ValueError("Model config is None! You MUST provide a model config in dict.")

        # For convenience
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_outputs = num_outputs

        # (0) Basic debug checks
        if self.cfg.ru_debugging:
            assert isinstance(self.cfg, VisionLanderPPOConfig), \
                f"model_config['custom_model_config'] is not VisionLanderPPOConfig. It is {type(self.cfg)}."
            action_size = action_space.shape[0]
            assert action_size == 3, f"action_size is not 3! It is {action_size}."
            assert num_outputs == 2 * action_size, \
                f"num_outputs is not 2 * action_size! It is {num_outputs} vs. {2*action_size}."

        # (1) Build networks
        if self.cfg.is_conv_shared:
            # 기존 처럼 한 벌의 encoder/embedding 만 사용
            self._build_shared_nets()
        else:
            # actor 와 critic 각각 별도 encoder/embedding 사용
            self._build_separate_nets()

        self._value_out = None
        self._current_obs = None  # value_function()에서 사용할 임시 저장공간

    def _build_encoder(self, input_channels: int) -> nn.Sequential:
        """공용으로 쓰는 함수: Conv encoder를 만드는 부분."""
        layers = []
        in_channels = input_channels
        for out_ch, k, s in zip(self.cfg.encoder_channels, self.cfg.kernel_sizes, self.cfg.strides):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_ch, kernel_size=k, stride=s))
            layers.append(nn.ReLU())
            in_channels = out_ch
        return nn.Sequential(*layers)

    def _get_conv_out_size(self, encoder: nn.Sequential, in_shape) -> int:
        """dummy input을 태워보고 Conv output shape을 구한다."""
        test_input = torch.zeros(1, *in_shape)  # (C, H, W)...
        conv_out = encoder(test_input)
        return int(np.prod(conv_out.shape[1:]))

    def _build_embedding(self, in_dim: int) -> nn.Sequential:
        """embedding 레이어: (conv_out + drone_state) -> embed_dim."""
        norm_layer = nn.LayerNorm(self.cfg.embed_dim) if self.cfg.use_layer_norm else nn.BatchNorm1d(self.cfg.embed_dim)
        return nn.Sequential(
            nn.Linear(in_dim, self.cfg.embed_dim),
            norm_layer,
            nn.Tanh()
        )

    def _build_fc_layers(self, in_size: int, hidden_sizes: List[int], out_size: int = None) -> nn.Sequential:
        """MLP 블럭을 만드는 유틸 함수"""
        layers = []
        prev_size = in_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        if out_size is not None:
            layers.append(nn.Linear(prev_size, out_size))
        return nn.Sequential(*layers)

    def _build_shared_nets(self):
        """is_shared_net=True 일 때(기존 로직)."""
        # -- encoder
        obs_space_images = self.obs_space.original_space["images"]
        input_channel_size = obs_space_images.shape[0]
        self.encoder = self._build_encoder(input_channel_size)
        self._conv_out_flattened_size = self._get_conv_out_size(
            self.encoder, (input_channel_size, obs_space_images.shape[1], obs_space_images.shape[2])
        )

        # -- embedding
        drone_state_size = self.obs_space.original_space["drone_state"].shape[0]
        encoder_plus_drone_dim = self._conv_out_flattened_size + drone_state_size
        self.embedding = self._build_embedding(encoder_plus_drone_dim)

        # -- policy net
        self.policy = self._build_fc_layers(
            in_size=self.cfg.embed_dim,
            hidden_sizes=self.cfg.policy_hidden_sizes,
            out_size=self.num_outputs
        )

        # -- value net
        self.critic = self._build_fc_layers(
            in_size=self.cfg.embed_dim,
            hidden_sizes=self.cfg.value_hidden_sizes,
            out_size=1
        )

    def _build_separate_nets(self):
        """is_shared_net=False 일 때, actor와 critic의 Conv-Encoder/Embedding까지 분리."""
        obs_space_images = self.obs_space.original_space["images"]
        input_channel_size = obs_space_images.shape[0]
        drone_state_size = self.obs_space.original_space["drone_state"].shape[0]

        # === Actor ===
        self.actor_encoder = self._build_encoder(input_channel_size)
        conv_out_actor = self._get_conv_out_size(
            self.actor_encoder, (input_channel_size, obs_space_images.shape[1], obs_space_images.shape[2])
        )
        self.actor_embedding = self._build_embedding(conv_out_actor + drone_state_size)
        self.actor_policy = self._build_fc_layers(
            in_size=self.cfg.embed_dim,
            hidden_sizes=self.cfg.policy_hidden_sizes,
            out_size=self.num_outputs
        )

        # === Critic ===
        self.critic_encoder = self._build_encoder(input_channel_size)
        conv_out_critic = self._get_conv_out_size(
            self.critic_encoder, (input_channel_size, obs_space_images.shape[1], obs_space_images.shape[2])
        )
        self.critic_embedding = self._build_embedding(conv_out_critic + drone_state_size)
        self.critic_value = self._build_fc_layers(
            in_size=self.cfg.embed_dim,
            hidden_sizes=self.cfg.value_hidden_sizes,
            out_size=1
        )

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # Choose the appropriate context manager based on your config flag.
        cm = torch.autograd.set_detect_anomaly(True) if self.cfg.use_anomaly_detection else contextlib.nullcontext()
        with cm:
            obs_dict = input_dict["obs"]
            images = obs_dict["images"]
            drone_state = obs_dict["drone_state"]
            batch_size = images.shape[0]

            # PPO에서 forward는 actor 쪽(정책 logits)만 우선 계산
            if self.cfg.eval_mode:
                # 평가 시에는 uint8 -> float 변환 등
                images = images.float() if images.dtype == torch.uint8 else images
            # 학습 시에는 보통 이미지가 이미 float32로 전처리되어 들어온다고 가정
            # images /= 255.0
            images = images.div(255.0)  # avoid in-place operation (may cause error for gradient calculation in PyTorch)

            if self.cfg.is_conv_shared:
                # === (공유 네트워크) 기존 로직 ===
                enc_out = self.encoder(images)
                enc_out_flat = enc_out.view(enc_out.shape[0], -1)
                emb_in = torch.cat([enc_out_flat, drone_state], dim=1)
                x_embd = self.embedding(emb_in)
                logits = self.policy(x_embd)
                # critic 값도 여기서 바로 구해둔다.
                self._value_out = self.critic(x_embd)
            else:
                # === (분리 네트워크) actor 경로만 계산 ===
                actor_enc_out = self.actor_encoder(images)
                actor_enc_out_flat = actor_enc_out.view(actor_enc_out.shape[0], -1)
                actor_emb_in = torch.cat([actor_enc_out_flat, drone_state], dim=1)
                actor_emb = self.actor_embedding(actor_emb_in)
                logits = self.actor_policy(actor_emb)
                # critic은 forward() 안에서 계산하지 않음
                self._value_out = None

                # critic forward에 필요한 obs를 저장해둔다(value_function()에서 사용)
                self._current_obs = obs_dict

            if self.cfg.ru_debugging and batch_size == 512 and not self.cfg.is_conv_shared:
                if torch.isnan(logits).any():
                    print(f"@@@@@@@@@@@@@ images (nan): {torch.isnan(images).sum()}")
                    print(f"@@@@@@@@@@@@@ drone_state (nan): {torch.isnan(drone_state).sum()}")
                    print(f"@@@@@@@@@@@@@ actor_enc_out (nan): {torch.isnan(actor_enc_out).sum()}")
                    print(f"@@@@@@@@@@@@@ actor_enc_out_flat (nan): {torch.isnan(actor_enc_out_flat).sum()}")
                    print(f"@@@@@@@@@@@@@ actor_emb_in (nan): {torch.isnan(actor_emb_in).sum()}")
                    print(f"@@@@@@@@@@@@@ actor_emb (nan): {torch.isnan(actor_emb).sum()}")
                    print(f"@@@@@@@@@@@@@ logits (nan): {torch.isnan(logits).sum(axis=0)}")
                    # print(f"@@@@@@@@@@@@@ self._value_out (nan): {torch.isnan(self._value_out).sum()}")
                    print("logits에서 NaN 발생!")
                if torch.isinf(logits).any():
                    raise ValueError("logits에서 Inf 발생!")

            return logits, state

    def value_function(self) -> TensorType:
        """RLlib이 정책 네트워크 forward 이후 호출하는 함수."""
        if self.cfg.is_conv_shared:
            # 공유 네트워크라면 forward()에서 self._value_out을 이미 계산해 둠
            assert self._value_out is not None, "value_function() called before forward()?"
            return self._value_out.squeeze(-1)
        else:
            # 분리 네트워크라면 여기서 critic 경로를 다시 계산
            assert self._current_obs is not None, "value_function() called before forward()?"
            images = self._current_obs["images"]
            drone_state = self._current_obs["drone_state"]

            if self.cfg.eval_mode:
                images = images.float() if images.dtype == torch.uint8 else images
            # images /= 255.0
            images = images.div(255.0)  # avoid in-place operation (may cause error for gradient calculation in PyTorch)

            critic_enc_out = self.critic_encoder(images)
            critic_enc_out_flat = critic_enc_out.view(critic_enc_out.shape[0], -1)
            critic_emb_in = torch.cat([critic_enc_out_flat, drone_state], dim=1)
            critic_emb = self.critic_embedding(critic_emb_in)
            value_out = self.critic_value(critic_emb)

            return value_out.squeeze(-1)

