import gym
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union

import ray
from ray.rllib.algorithms.sac.sac_torch_policy import (
    SACTorchPolicy,
    # actor_critic_loss,
    build_sac_model_and_action_dist,
    stats,
    optimizer_fn,
    setup_late_mixins,
    action_distribution_fn,
    _get_dist_class,
    postprocess_trajectory,
    validate_spaces,
)
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    huber_loss,
)
from ray.rllib.utils.typing import TensorType, AlgorithmConfigDict
from ray.rllib.algorithms.sac.sac_torch_policy import ComputeTDErrorMixin

# import gym_pybullet_drones.rl.drq.drq
#
from gym_pybullet_drones.rl.drq.drq_torch_model import DrQTorchModel
from ray.rllib.models import MODEL_DEFAULTS
import copy

# todos:
# - [o] Apply augmentation to the policy model as well <-- done in the model (i.e. drq_torch_model.py)
# - [ ] Use same augmented images for both Q and policy model
def drq_actor_critic_loss(
        policy: Policy,
        model: DrQTorchModel,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Optimized actor critic loss with DrQ regularization.

    Implements the Data-regularized Q (DrQ) loss function with:
    1. Image augmentation via random shifts
    2. Averaging the Q target over K augmentations (equation 1 in paper)
    3. Averaging the Q function over M augmentations (equation 3 in paper)

    Args:
        policy: Policy to train
        model: Model to train
        dist_class: Action distribution class
        train_batch: Training data

    Returns:
        Loss tuple containing actor, critic, and alpha losses
    """
    # Get target model
    target_model = policy.target_models[model]

    # DrQ parameters
    k = policy.config.get("num_q_target_augmentations", 2)  # K in the paper
    m = policy.config.get("num_q_augmentations", 2)         # M in the paper

    # For debugging only
    deterministic = policy.config["_deterministic_loss"]

    # Process observations once
    obs_t = train_batch[SampleBatch.CUR_OBS]
    obs_tp1 = train_batch[SampleBatch.NEXT_OBS]

    # Compute model outputs - single forward pass per network
    model_out_t, _ = model(
        SampleBatch(obs=obs_t, _is_training=True), [], None
    )
    model_out_tp1, _ = model(
        SampleBatch(obs=obs_tp1, _is_training=True), [], None
    )
    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=obs_tp1, _is_training=True), [], None
    )

    # Alpha value for entropy regularization
    alpha = torch.exp(model.log_alpha)

    # Policy outputs - compute once and reuse
    action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
    action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)

    twin_q_t = None
    twin_q_t_selected = None
    twin_td_error = None

    # Discrete action space case
    if model.discrete:
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)

        # Q-values with augmentation - let the model handle the repetition and averaging
        q_t, _ = model.get_q_values(model_out_t, num_augmentations=m)

        # Twin Q-values if enabled
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t, num_augmentations=m)

        # Target Q-values with augmentation
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, num_augmentations=k)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, num_augmentations=k
            )
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_tp1 -= alpha * log_pis_tp1

        # Actually selected Q-values (from the actions batch)
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)

        # Discrete case: "Best" means weighted by the policy (prob) outputs
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best
    # Continuous action space case
    else:
        # Sample single actions from distribution - reuse dist objects
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)

        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

        # Q-values with augmentation
        actions = train_batch[SampleBatch.ACTIONS]
        q_t, _ = model.get_q_values(model_out_t, actions, num_augmentations=m)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t, actions, num_augmentations=m)

        # Q-values for current policy in given current state
        # No augmentation needed here as it's for actor loss
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

        # Target q network evaluation with augmentation
        q_tp1, _ = target_model.get_q_values(
            target_model_out_tp1, policy_tp1, num_augmentations=k
        )
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1, num_augmentations=k
            )
            # Take min over both twin-NNs
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)

        q_tp1 -= alpha * log_pis_tp1

        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

    # Compute RHS of bellman equation - detached to stop gradients
    q_t_selected_target = (
            train_batch[SampleBatch.REWARDS]
            + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    # Compute the TD-error (potentially clipped)
    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    # Calculate critic loss - reuse weights tensor
    weights = train_batch[PRIO_WEIGHTS]
    critic_loss = [torch.mean(weights * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(torch.mean(weights * huber_loss(twin_td_error)))

    print(f"policy.config['twin_q']: {policy.config['twin_q']}")

    if policy.config["twin_q"]:
        assert twin_q_t is not None and twin_q_t_selected is not None and twin_td_error is not None, (
            f"Twin Q-values are not None, but they should be. Your policy.config.twin_q=={policy.config['twin_q']} "
        )
    else:
        assert twin_q_t is None or twin_q_t_selected is None or twin_td_error is None, (
            f"Twin Q-values are Not None, but they should be. Your policy.config.twin_q=={policy.config['twin_q']} "
        )

    # Alpha- and actor losses
    if model.discrete:
        # Pre-compute factor for better efficiency
        entropy_factor = -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        weighted_log_alpha_loss = policy_t.detach() * entropy_factor
        # Sum up weighted terms and mean over all batch items
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))

        # Actor loss - precompute common factor
        q_minus_entropy = q_t.detach() - alpha.detach() * log_pis_t
        actor_loss = torch.mean(torch.sum(policy_t * -q_minus_entropy, dim=-1))
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    # Store values for stats function
    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss
    model.tower_stats["td_error"] = td_error

    # Return all loss terms
    return tuple([actor_loss] + critic_loss + [alpha_loss])

def build_drq_model_and_action_dist(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Creates a DrQTorchModel and action distribution for DrQTorchPolicy.

    Args:
        policy: Policy instance
        obs_space: Observation space
        action_space: Action space
        config: Algorithm configuration

    Returns:
        Tuple of model and action distribution class
    """
    shift_pad = config.get("shift_pad", 4)
    model = build_drq_model(policy, obs_space, action_space, config, shift_pad)
    action_dist_class = _get_dist_class(policy, config, action_space)

    return model, action_dist_class

def build_drq_model(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: AlgorithmConfigDict,
        shift_pad: int = 4,
) -> ModelV2:
    """Constructs the necessary ModelV2 for the Policy and returns it.

    Args:
        policy: The Torch|TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The DrQ trainer's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy.
            Note: An additional target model will be created in this function and assigned to `policy.target_model`.
    """
    policy_model_config = copy.deepcopy(MODEL_DEFAULTS)
    policy_model_config.update(config["policy_model_config"])
    policy_model_config.update({"_disable_preprocessor_api": config["_disable_preprocessor_api"]})
    q_model_config = copy.deepcopy(MODEL_DEFAULTS)
    q_model_config.update(config["q_model_config"])
    q_model_config.update({"_disable_preprocessor_api": config["_disable_preprocessor_api"]})

    num_outputs = action_space.shape[0]
    model = DrQTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name="drq_model",
        # policy_model_config=config["policy_model_config"],
        # q_model_config=config["q_model_config"],
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        shift_pad=shift_pad,
    )
    # Create an exact copy of the model and store it in `policy.target_model`.
    # This will be used for tau-synched Q-target models that run behind the
    # actual Q-networks and are used for target q-value calculations in the
    # loss terms.
    policy.target_model = DrQTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        name="drq_model",
        # policy_model_config=config["policy_model_config"],
        # q_model_config=config["q_model_config"],
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
        shift_pad=shift_pad,
    )
    return model

# Build the DrQTorchPolicy class
DrQTorchPolicy = build_policy_class(
    name="DrQTorchPolicy",
    framework="torch",
    loss_fn=drq_actor_critic_loss,
    get_default_config=lambda: ray.rllib.algorithms.sac.sac.DEFAULT_CONFIG,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_drq_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=action_distribution_fn,
)
