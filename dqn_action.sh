cd /root/BitFlip/DQN2
CUDA_VISIBLE_DEVICES=1 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn step_is_fast=true