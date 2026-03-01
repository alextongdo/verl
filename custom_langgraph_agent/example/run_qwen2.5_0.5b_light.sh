#!/usr/bin/env bash
#SBATCH --job-name=rl-langgraph-0p5B-light
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -xeuo pipefail

export GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${GPUS_PER_NODE:-1}}
NNODES=${SLURM_JOB_NUM_NODES:-${NNODES:-1}}
export NNODES
export RAY_NUM_NODES=$NNODES

TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
if [ "$TOTAL_GPUS" -lt 1 ]; then
  echo "Error: at least 1 GPU is required, detected $TOTAL_GPUS." >&2
  exit 1
fi

echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

model_path=${model_path:-$DATA_ROOT/model/Qwen2.5-0.5B-Instruct}
if [ ! -d "$model_path" ]; then
  model_path=Qwen/Qwen2.5-0.5B-Instruct
fi

train_files=$DATA_ROOT/data/math_expression_tool/train.parquet
test_files=$DATA_ROOT/data/math_expression_tool/test.parquet
agent_loop_config_path=recipe/custom_langgraph_agent/example/agent.yaml

project_name=math_expression_tool
experiment_name=qwen2.5-0.5b-light
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=1024
max_response_length=1024
actor_lr=1e-6

train_batch_size=16
ppo_mini_batch_size=4
n_resp_per_prompt=2
n_resp_per_prompt_val=1

export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

infer_tp=1
train_sp=1
offload=false

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=true \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.log_val_generations=50 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=-1 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 "$@"
