# BitFlipping with Deep Q-Network
## Abstract
The purpose of this experiment report is to investigate and evaluate the application of Deep Q-Network(DQN) in the task of BitFlipping problem. BitFlipping problem is a classical problem in Computer Science. The conventional approach typically involves enumerating over a space of size $2^n$, leading to a high time complexity. However, Reinforcement Learning methods can explore this vast search space by using rewards from the constructed environment to guide the agent towards optimizing for the highest expected rewards. This approach helps avoid unnecessary enumerations. In this study, I attempted to address this issue using the DQN algorithm from reinforcement learning. Besides, I designed DQN\_with\_GOAL to enhance the generalization of goals, leveraged hindsight experience replay tricks to mitigate the sparse reward issue and give an analysis of reward shaping.

## BitFlipping environment
**State space**: $\mathcal{S} = \{0,1\}^n$, $n \in \{1,...,50\}$

**Action space**: $\mathcal{A} = \{0,1,...,n-1\}$ for some integer n in which executing the i-th action flips the i-th bit of the state.

**Reward**: The reward is 0 if the final sequence generated is not equal to the target sequence, and is -1 otherwise.

For each episode, a target is generated randomly.

## Installation and Dependencies
* python >= 3.9
* torch >= 2.0.0

Other dependencies can be installed using the following command:
```
conda create -n ling python=3.9
pip install -r requirements.txt
```
## Usage
* Run the training script by DQN
```
cd DQN
python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn
```
* Run the training script by DQN_with_GOAL
```
cd DQNwithGOAL
python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_g
```
* Run the training script by DQN_with_HER
```
cd DQNwithGOAL
python main.py length=8 epsilon=0.9 delta_epsilon=1e-5 target_update=50 use_wandb=false reward_success=0 reward_fail=-1 exp_name=dqn_with_her
```
## Reward Settings
BitFlipping environments consist of 3 reward settings: Binary, euclidean and 'Step by Step'.

**Binary** means that the reward only contains 2 integers: **success_reward** and **fail_reward**.
The reward is **success_reward** if the final sequence generated is equal to the target sequence, and is
**fail_reward** otherwise. **success_reward** and **fail_reward** are 2 hyperparameters.

**euclidean** means that $r(s_t,a_t,g) = -||s_t-g_t||_2$

**Step by Step** means that 
$$
\text{reward} = \begin{cases} 
    0 & \text{if } \text{state[action]} = \text{goal[action]} \\
    -1 & \text{otherwise}
\end{cases}
$$

if you want to use **Binary**, then
```
python main.py env_reward_type=default
```
if you want to use **euclidean**, then
```
python main.py env_reward_type=euclidean
```
if you want to use **Step by Step**, then
```
python main.py env_reward_type=idx
```
## Reference
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf)
- [Hands on RL](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95/)
