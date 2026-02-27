#!/bin/bash

# Define the grids
MODELS=("Qwen/Qwen2.5-0.5B-Instruct" "HuggingFaceTB/SmolLM2-135M")
EPOCHS=(1 2)
MICRO_BATCHES=(4 8 16)

for MODEL in "${MODELS[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        for MB in "${MICRO_BATCHES[@]}"; do
            MODEL_SHORT=$(echo $MODEL | cut -d'/' -f2)
            RUN_NAME="rl_ppo_${MODEL_SHORT}_ep${EPOCH}_mb${MB}"
            
            echo "------------------------------------------------"
            echo "Starting Run: $RUN_NAME"
            echo "------------------------------------------------"

            PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
                global_profiler.global_tool_config.nsys.worker_nsight_options.trace=\'cuda,nvtx,osrt\' \
                global_profiler.global_tool_config.nsys.worker_nsight_options.cuda-memory-usage=\'true\' \
                global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi \
                global_profiler.steps=[1,2,5,10] \
                +global_profiler.discrete=False \
                global_profiler.tool=nsys \
                actor_rollout_ref.actor.profiler.enable=True \
                actor_rollout_ref.actor.profiler.all_ranks=True \
                critic.profiler.enable=True \
                critic.profiler.all_ranks=True \
                +reward_model.profiler.enable=True \
                +reward_model.profiler.all_ranks=True \
                data.train_files=$HOME/data/gsm8k/train.parquet \
                data.val_files=$HOME/data/gsm8k/test.parquet \
                data.train_batch_size=256 \
                data.max_prompt_length=512 \
                data.max_response_length=512 \
                actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
                actor_rollout_ref.actor.optim.lr=1e-6 \
                actor_rollout_ref.actor.ppo_mini_batch_size=64 \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
                actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
                actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
                critic.optim.lr=1e-5 \
                critic.model.path=$MODEL \
                critic.ppo_micro_batch_size_per_gpu=$MB \
                critic.ppo_epochs=$EPOCH \
                algorithm.kl_ctrl.kl_coef=0.001 \
                trainer.logger=[console,wandb] \
                trainer.experiment_name=$RUN_NAME \
                trainer.val_before_train=False \
                trainer.n_gpus_per_node=1 \
                trainer.nnodes=1 \
                trainer.save_freq=10 \
                trainer.test_freq=10 \
                trainer.total_epochs=1 \
                trainer.total_training_steps=10

                hf upload randomath/rl_ppo_profiles /tmp/ray/session_latest/logs/ ./$RUN_NAME --repo-type=dataset
        done
    done
done
