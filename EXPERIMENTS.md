```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

wandb login  # get API key from 

# ppo_colocated_baseline
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.profiler.enable=True \
    critic.profiler.all_ranks=False \
    critic.profiler.ranks=[0] \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=6 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=cse291p-wi26 \
    trainer.experiment_name=ppo_colocated_baseline \
    global_profiler.steps="[2,3,4]" \
    global_profiler.tool=nsys \
    global_profiler.global_tool_config.nsys.discrete=False \
    'global_profiler.global_tool_config.nsys.worker_nsight_options.trace="cuda,nvtx,osrt"' \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi" \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=repeat" \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.force-overwrite='"true"' \
    2>&1 | tee ppo_colocated_baseline.log

# grpo_colocated_tail
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=cse291p-wi26 \
    trainer.experiment_name=grpo_colocated_tail \
    trainer.rollout_data_dir=/workspace/rollout_logs/grpo_colocated_tail \
    global_profiler.steps="[2,3,4]" \
    global_profiler.tool=nsys \
    global_profiler.global_tool_config.nsys.discrete=False \
    'global_profiler.global_tool_config.nsys.worker_nsight_options.trace="cuda,nvtx"' \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi" \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=repeat" \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.force-overwrite='"true"' \
    2>&1 | tee grpo_colocated_tail.log





PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    actor_rollout_ref.actor.profiler.ranks=[0] \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.all_ranks=False \
    actor_rollout_ref.rollout.profiler.ranks=[0] \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=cse291p-wi26 \
    trainer.experiment_name=grpo_disaggregated_7b \
    trainer.rollout_data_dir=/workspace/rollout_logs/grpo_disaggregated_7b \
    global_profiler.steps="[3]" \
    global_profiler.tool=nsys \
    global_profiler.global_tool_config.nsys.discrete=False \
    'global_profiler.global_tool_config.nsys.worker_nsight_options.trace="cuda,nvtx"' \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi" \
    "global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end=repeat" \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.force-overwrite='"true"' \
    2>&1 | tee grpo_disaggregated_7b.log
```

cp -r /tmp/ray/3/logs/nsight/ /workspace/nsight/ && chmod -R 777 /workspace/nsight/
chmod -R 777 /workspace/rollout_logs/grpo_colocated_tail





actor_rollout_ref.actor.strategy=fsdp2 \
  rollout.total_rollout_steps=3840 \
```bash
PYTHONUNBUFFERED=1 python3 -m verl.experimental.fully_async_policy.fully_async_main \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \

  data.train_batch_size=0 \
  data.gen_batch_size=1 \

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
  actor_rollout_ref.rollout.n=8 \

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \

  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.mode=async \

  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \

  rollout.nnodes=1 \
  rollout.n_gpus_per_node=1 \

  async_training.staleness_threshold=0 \
  async_training.require_batches=1 \
  async_training.trigger_parameter_sync_step=4 \
  async_training.partial_rollout=False \

  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=cse291p-wi26 \
  trainer.experiment_name=grpo_disaggreated_baseline \
  trainer.val_before_train=False \
  trainer.total_training_steps=5 \

  global_profiler.steps="[2]" \
  global_profiler.save_path=./profile \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=False \
  actor_rollout_ref.actor.profiler.ranks=[0] \
  actor_rollout_ref.actor.profiler.tool_config.torch.discrete=True \
  actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda] \
  actor_rollout_ref.rollout.profiler.enable=True \
  actor_rollout_ref.rollout.profiler.all_ranks=False \
  actor_rollout_ref.rollout.profiler.ranks=[0] \
  actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
  actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cuda] \

  2>&1 | tee grpo_disaggreated_baseline.log
```
chmod -R 777 ./profile/