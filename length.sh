cd /root/BitFlip/DQN
CUDA_VISIBLE_DEVICES=0 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_1_0
