set -x


python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.rollout.n=4 \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.rollout.agent.num_workers=1 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 algorithm.use_kl_in_reward=False \
 trainer.logger='["console","wandb"]' \
 trainer.project_name="PPO" \
 trainer.experiment_name="GRPO_1" \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_training_steps=9 \
 trainer.resume_mode=disable \
 global_profiler.tool=nsys \
 global_profiler.steps=[5,6,7,8] \
 global_profiler.profile_continuous_steps=True \
 global_profiler.global_tool_config.nsys.discrete=False \
 actor_rollout_ref.actor.profiler.enable=True \
 actor_rollout_ref.actor.profiler.all_ranks=False \
 critic.profiler.enable=False \
 trainer.total_epochs=1 2>&1 | tee verl_grpo1.log 