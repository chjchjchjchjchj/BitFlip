cd /root/BitFlip/DQNwithHer
for length in {4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40}; do
    CUDA_VISIBLE_DEVICES=3 python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_her
done