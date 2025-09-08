#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
pip install latex2sympy2_extended
pip install word2number
pip install GPUtil

# ray start --head --node-ip-address=$(hostname -I | awk '{print $1}') --port=6379

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS


DeepScaleR_train_path=/DeepScaleR-Preview-Dataset/train.parquet


math500_test_path=/MATH-500/test.parquet
aime24_test_path=/aime_2024_test.parquet
aime25_test_path=/aime_2025_test.parquet


train_files="['$DeepScaleR_train_path']"
test_files="['$math500_test_path', '$aime24_test_path', '$aime25_test_path']"


python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=/.../VSRM-EFFICIENT-LRMS/verl/utils/reward_score/val_reward.py \
    custom_reward_function.name=compute_math_val \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=768 \
    data.max_response_length=7424 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/huggingface.co/agentica-org/DeepScaleR-1.5B-Preview \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n2=5 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/huggingface.co/agentica-org/DeepScaleR-1.5B-Preview \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='step_wise_reward_ppo' \
    trainer.experiment_name='exp8' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 $@ \


