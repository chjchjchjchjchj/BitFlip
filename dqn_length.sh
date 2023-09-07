cd /root/BitFlip/DQN
for length in {1..20}; do
    CUDA_VISIBLE_DEVICES=1 python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn
done