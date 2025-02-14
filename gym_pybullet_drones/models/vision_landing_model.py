import copy  # Remove this if not used (later)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
# typing
from typing import List, Union, Dict
from dataclasses import dataclass, field
# torch
import torch
import torch.nn as nn
# custom modules
# from yours


class VisionLanderPPOConfig:
    ru_debugging: bool = False
    is_shared_net: bool= False
    # 필요한 다른 하이퍼파라미터들도 추가

    # dataclass에서 dict로 변환하기
    def to_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "ru_debugging": self.ru_debugging,
            # ...
        }


class VisionLanderPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # (0) Get model config
        # (0-0) Load model config
        if model_config is not None:
            self.cfg: VisionLanderPPOConfig = model_config["custom_model_config"]
        else:
            raise ValueError("Model config is None! Please you MUST provide a model config in dict.")
        if self.cfg.ru_debugging:
            assert isinstance(self.cfg, VisionLanderPPOConfig), \
                f"model_config['custom_model_config'] is not VisionLanderPPOConfig. It is {type(self.cfg)}."

        # (0-1) Validate the model config (set default values if not provided)
        #   Here, the debug flag does not skip the validation as config error may be caused by human error.

        # (1) Define model modules (or layers)
        # (1-0) Encoder
        # (1-1) policy modules
        # (1-2) value modules (and branch if shared)

        # (2) Define policy network
        self.actor = NotImplementedError

        # (3) Define value network
        self.value_branch = NotImplementedError

    def _validate_config(
            self,
            config: Dict[str, Union[str, int, float, bool]]
    ) -> Dict[str, Union[str, int, float, bool]]:
        validated_config = copy.deepcopy(config)

        return NotImplementedError

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs_dict = input_dict["obs"]

        x = obs_dict

        # shape: ...
        return x, state

    def value_function(self) -> TensorType:
        assert self.values is not None, "self.values is None! Please check if the value network is defined."
        assert self.value.dim() == 2, f"self.value.dim() is not 2! It is {self.value.dim()}."
        value = self.value_branch(self.values).squeeze(1)  # shape: (batch_size,)
        return value

