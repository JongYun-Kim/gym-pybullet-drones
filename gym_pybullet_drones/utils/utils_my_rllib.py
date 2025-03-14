from ray.rllib.algorithms.callbacks import MultiCallbacks
import torch.nn as nn

def orthogonal_init(module: nn.Module):
    """
    Usage:
        model.apply(orthogonal_init)
    """
    # 입력 validation: weight와 bias 속성이 있는지 확인하는 간단한 검사를 추가할 수 있습니다.
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.orthogonal_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)

def create_multi_callbacks_from_classes(callback_classes):
    """
    **MultiCallbacks 객체를 만들어 주는 헬퍼 함수; config 에 저장할 string 리스트도 반환**

    Parameters:
        callback_classes: 콜백 클래스들의 리스트
    Returns:
        MultiCallbacks 객체,
        Callback class name list

    Example:
        callback_classes = [MyCallback1, MyCallback2]
        multi_callbacks, callback_names = create_multi_callbacks_from_classes(callback_classes)
        config = {
            "env": "CartPole-v0",
            ...
            "callbacks": multi_callbacks,
            "callbacks_names": callback_names  # Used for logging in params.json
            ...
        }
    """
    if len(callback_classes) == 0:
        print(" @@ No callback classes found; returning None and empty list.")
        return None, []
    elif len(callback_classes) == 1:
        print(f" @@ Only one callback class found; returning the class and its name. ({callback_classes[0].__name__})")
        return callback_classes[0], [callback_classes[0].__name__]
    else:
        print(f" @@ Multiple callback classes found; returning MultiCallbacks object and their names.")
        print(f" @@ Callback classes: {[cls.__name__ for cls in callback_classes]}")
        return MultiCallbacks(callback_classes), [cls.__name__ for cls in callback_classes]


