import sys
sys.path.append("/root/BitFlip")
import gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import collections
import random
import torch.nn.functional as F
import ipdb
import matplotlib.pyplot as plt
import rl_utils
import hydra
import os
import wandb
import omegaconf
from DQN import *
from my_env import BitFlip
import json
from tensorboardX import SummaryWriter

def save_results_to_json(log_episodes, win_rate, epsilon_array):
    if not os.path.exists('results'):
        os.makedirs('results')
    results = {
        "log_episodes": log_episodes,
        "win_rate": win_rate,
        "epsilon_array": epsilon_array
    }
    with open('results.json', 'w') as json_file:
        json.dump(results, json_file)

def set_seed(seed, env):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

@hydra.main(config_path="configs", config_name="defaults", version_base="1.1")
def main(args):
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    wandb_project = "BitFlip" #@param {"type": "string"}
    wandb_run_name = args.exp_name
    # ipdb.set_trace()
    args_dict = omegaconf.OmegaConf.to_container(args, resolve=True)
    # wandb.init(project=wandb_project, name=wandb_run_name, job_type="baseline-train", config=args)
    # wandb.init(project=wandb_project, name=wandb_run_name, config=args)
    wandb.init(project=wandb_project, name=wandb_run_name, config=args_dict)
    some_threshold = 0
    length = args.length
    lr = args.lr
    num_episodes = args.num_episodes
    hidden_dim = args.hidden_dim
    gamma = args.gamma
    epsilon = args.epsilon
    target_update = args.target_update
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    eval = args.eval
    log_frequency = args.log_frequency
    # max_steps = args.length if args.max_steps > args.length else args.max_steps
    max_steps = args.length
    minimal_epsilon = args.minimal_epsilon
    delta_epsilon = args.delta_epsilon
    reward_type = args.env_reward_type
    reward_success = args.reward_success
    reward_fail = args.reward_fail
    model_name = args.model_name
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_dir = "logs/"
    writer = SummaryWriter(log_dir=log_dir, comment=f"bits={args.length},model={model_name},reward={(reward_success,reward_fail)},init_e={args.epsilon}")
    # device = torch.device('cpu')
    env_name = "BitFlip"
    env = BitFlip(length=length, reward_type=reward_type, reward_success=reward_success, reward_fail=reward_fail)


    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)

    set_seed(10, env=env)
    minimal_size = args.minimal_size
    assert minimal_size >= batch_size
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')
    agent = DQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, learning_rate=lr, gamma=gamma, epsilon=epsilon, target_update=target_update, device=device, length=length, policy_type="goal", checkpoint_dir=checkpoint_dir, minimal_epsilon=minimal_epsilon, delta_epsilon=delta_epsilon)

    win_rate = []
    log_episodes = []
    epsilon_array = []

    success = 0
    for episode in tqdm(range(num_episodes)):
        episode_return = 0
        state, goal = env.reset()
        done = False
        for step in range(max_steps):
            if not done:
                action = agent.take_action(state=state, goal=goal)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done, goal)
                state = next_state
                episode_return += reward
                if not args.eval:
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d, b_g = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d,
                            'goals': b_g
                        }
                        agent.learn(transition_dict)
                if done:
                    success += 1
        
        if episode % log_frequency == 0:
            if len(win_rate) > 0 and (success / log_frequency) > win_rate[-1]:
                agent.save_model()
            win_rate.append(success / log_frequency)
            log_episodes.append(episode)
            epsilon_array.append(agent.epsilon)
            print(f"win_rate={win_rate}")
            print(f"epsilon_array={epsilon_array}")
            wandb.log({
                "win_rate": success / log_frequency,
                "episode": episode,
                "epsilon": agent.epsilon,
            })
            writer.add_scalar('win_rate', success / log_frequency, global_step=episode)
            writer.add_scalar('epsilon', agent.epsilon, global_step=episode)
            save_results_to_json(log_episodes, win_rate, epsilon_array)
            success = 0
    
    figure = plt.figure()
    plt.title(f"DQNwithGOAL in {length} bits, minimal_size={minimal_size}")
    plt.ylabel("Win Rate")
    plt.xlabel("Episodes")
    plt.ylim([0, 1.1])
    plt.plot(log_episodes, win_rate)
    plt.show()

    save_plot_dir = os.path.join(os.getcwd(), 'plots')
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    save_win_rate_path = os.path.join(save_plot_dir, 'win_rate.pdf')
    plt.savefig(save_win_rate_path)

                    

if __name__ == "__main__":
    main()   