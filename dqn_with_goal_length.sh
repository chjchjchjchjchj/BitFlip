# cd /root/BitFlip/DQNwithGOAL
# for length in {4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40}; do
#     CUDA_VISIBLE_DEVICES=0 python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_g
# done


# cd /root/BitFlip/DQNwithGOAL
# for length in {4,8,12,16,20,24,28,32,36,40}; do
#     CUDA_VISIBLE_DEVICES=2 python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_g env_reward_type=idx
# done

# cd /root/BitFlip/DQNwithGOAL
# for length in {6,10,14,18,22,26,30,34,38}; do
#     CUDA_VISIBLE_DEVICES=3 python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_g env_reward_type=idx
# done

python main.py length=$length epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_g