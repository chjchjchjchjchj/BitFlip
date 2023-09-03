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

def set_seed(seed, env):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

@hydra.main(config_path="configs", config_name="defaults", version_base="1.1")
def main(args):
    if not args.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    wandb_project = "BitFlip_dqn_with_goal" #@param {"type": "string"}
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
    max_steps = args.max_steps if args.max_steps > args.length else args.length

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    env_name = "BitFlip"
    env = BitFlip(length=length)

    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)

    set_seed(10, env=env)
    minimal_size = 500
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')
    agent = DQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, learning_rate=lr, gamma=gamma, epsilon=epsilon, target_update=target_update, device=device, length=length, policy_type="goal", checkpoint_dir=checkpoint_dir)

    win_rate = []
    log_episodes = []
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
            success = 0
            wandb.log({
                "win_rate": success / log_frequency,
                "episode": episode
            })
    
    plt.title(f"DQN in {length} bits")
    plt.ylabel("Win Rate")
    plt.xlabel("Episodes")
    plt.ylim([0, 1.3])

    save_plot_dir = os.path.join(os.getcwd(), 'plots')
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    save_win_rate_path = os.path.join(save_plot_dir, 'win_rate.pdf')
    plt.savefig(save_win_rate_path)

                        
    # return_list = []
    # for i in range(10):
    #     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
    #         for i_episode in range(int(num_episodes / 10)):
    #             episode_return = 0
    #             state, goal = env.reset()
    #             # state = env.reset()
    #             done = False
    #             while not done:
    #                 action = agent.take_action(state=state, goal=goal)
    #                 # next_state, reward, done = env.step(action)
    #                 next_state, reward, done, _ = env.step(action)
    #                 replay_buffer.add(state, action, reward, next_state, done)
    #                 state = next_state
    #                 episode_return += reward

    #                 if replay_buffer.size() > minimal_size:
    #                     b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
    #                     transition_dict = {
    #                         'states': b_s,
    #                         'actions': b_a,
    #                         'next_states': b_ns,
    #                         'rewards': b_r,
    #                         'dones': b_d
    #                     }
    #                     agent.update(transition_dict)

    #             wandb.log({
    #                 "return": episode_return
    #             })

    #             return_list.append(episode_return)
    #             if (i_episode + 1) % 10 == 0:
    #                 pbar.set_postfix(
    #                     {
    #                         'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
    #                         'return': '%.3f' % np.mean(return_list[-10:])
    #                     }
    #                 )
    #                 pbar.update(1)
    #                 if episode_return > some_threshold:
    #                     save_model(agent.q_net, i_episode, args.model_save_dir)


    #     episodes_list = list(range(len(return_list)))
    #     plt.plot(episodes_list, return_list)
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Returns')
    #     plt.title('DQN on {}'.format(env_name))
    #     plt.savefig('DQN on {}(bf).pdf'.format(env_name))
    #     plt.show()

    #     mv_return = rl_utils.moving_average(return_list, 9)
    #     plt.plot(episodes_list, mv_return)
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Returns')
    #     plt.title('DQN on {}'.format(env_name))
    #     plt.savefig('DQN on {}(m).pdf'.format(env_name))
    #     plt.show()

if __name__ == "__main__":
    main()   