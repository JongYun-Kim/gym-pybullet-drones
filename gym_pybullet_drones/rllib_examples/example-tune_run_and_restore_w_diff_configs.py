"""
In best practices, I think, it might be the best to run them with an 'algorithm' object (trainable in old APIs)...
But, I like tune.run() ! You can also test hyperparameters-tuning scenarios based on this! Fingers crossed, mate!

The example is based on the following:
- First, train a PPO agent(algorithm) on CartPole-v1 for 4 iterations.
- Then, load the checkpoint of the 3rd iteration
- And then, resume training for 6 iterations with a different learning rate (1e-3 -> 5e-4).
- The second training starts from the 3rd iteration (i.e. the first result is from the 4th iteration).
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

ray.init(local_mode=True)

# Step 1: 처음 학습을 진행 (간단한 환경 사용)
initial_stop_iters = 4

analysis = tune.run(
    "PPO",
    stop={"training_iteration": initial_stop_iters},
    checkpoint_freq=1,  # 매 iteration마다 checkpoint 저장
    config={
        "env": "CartPole-v1",
        "framework": "torch",
        "lr": 1e-3,
        "num_gpus": 0,
        "num_workers": 0,  # 로컬에서 빠르게 테스트
    },
    local_dir="/home/ray/temps/debugging_only/restore_test",
)

# 가장 최근의 체크포인트 가져오기 (여기서는 iteration=3 것을 골라봄)
checkpoint_path = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial(), metric="episode_reward_mean"
)[2][0]  # 3번째 iteration 체크포인트

print(f"선택된 체크포인트: {checkpoint_path}")

# Step 2: 위 체크포인트를 로드하고, 다른 learning rate와 iteration을 적용하여 재학습 수행
new_stop_iters = 6
new_lr = 5e-4

# 체크포인트 로드 후 재학습 (restore 옵션 사용)
analysis_resume = tune.run(
    "PPO",
    stop={"training_iteration": new_stop_iters},  # 더 큰 값
    restore=checkpoint_path,
    checkpoint_freq=1,
    config={
        "env": "CartPole-v1",
        "framework": "torch",
        "lr": new_lr,  # 바뀐 learning rate
        "num_gpus": 0,
        "num_workers": 0,
    },
    local_dir="/home/ray/temps/debugging_only/resume_test",
)

print("재학습 완료")
print("pause here")

ray.shutdown()
