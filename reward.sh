# Reward的设计

cd /root/BitFlip/DQN
CUDA_VISIBLE_DEVICES=0 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_1_0
CUDA_VISIBLE_DEVICES=1 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_100_0
CUDA_VISIBLE_DEVICES=2 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_0_-1
CUDA_VISIBLE_DEVICES=3 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_1_-1

cd /root/BitFlip/DQNwithGOAL
CUDA_VISIBLE_DEVICES=0 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_g_1_0
CUDA_VISIBLE_DEVICES=1 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_g_100_0
CUDA_VISIBLE_DEVICES=2 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_g_0_-1
CUDA_VISIBLE_DEVICES=3 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_g_1_-1



cd /root/BitFlip/DQNwithHer
CUDA_VISIBLE_DEVICES=0 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_with_her_1_0
CUDA_VISIBLE_DEVICES=1 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_with_her_100_0
CUDA_VISIBLE_DEVICES=2 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_with_her_0_-1
CUDA_VISIBLE_DEVICES=3 python main.py length=4 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_with_her_1_-1




cd /root/BitFlip/DQN
CUDA_VISIBLE_DEVICES=0 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_1_0_8bits
CUDA_VISIBLE_DEVICES=1 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_100_0_8bits
CUDA_VISIBLE_DEVICES=2 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_0_-1_8bits
CUDA_VISIBLE_DEVICES=3 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_1_-1_8bits

cd /root/BitFlip/DQNwithGOAL
CUDA_VISIBLE_DEVICES=0 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_g_1_0_8bits
CUDA_VISIBLE_DEVICES=1 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_g_100_0_8bits
CUDA_VISIBLE_DEVICES=2 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_g_0_-1_8bits
CUDA_VISIBLE_DEVICES=3 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_g_1_-1_8bits

cd /root/BitFlip/DQNwithHer
CUDA_VISIBLE_DEVICES=0 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=0 exp_name=dqn_with_her_1_0_8bits
CUDA_VISIBLE_DEVICES=1 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=100 reward_fail=0 exp_name=dqn_with_her_100_0_8bits
CUDA_VISIBLE_DEVICES=2 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=0 reward_fail=-1 exp_name=dqn_with_her_0_-1_8bits
CUDA_VISIBLE_DEVICES=3 python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=true reward_success=1 reward_fail=-1 exp_name=dqn_with_her_1_-1_8bits


