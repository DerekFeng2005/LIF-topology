import argparse
import datetime
# import gym
import gymnasium as gym
import numpy as np
import itertools
import torch
import time
import math
from sac import SAC
from matplotlib import animation
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
import os

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="InvertedPendulum-v4",
                    help='Mujoco Gym environment (default: InvertedPendulum-v4)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: False)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--num_steps', type=int, default=101100, metavar='N',
                    help='maximum number of steps (default: 101100)')
parser.add_argument('--hidden_size', type=int, default=32, metavar='N',
                    help='hidden size (default: 32)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                    help='Steps sampling random actions (default: 2000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--model_name', default="LIF_HH",
                    help='choose model (choice: LIF, HH, LIF_HH, LIF_1_3, LIF_2_2, LIF_1_2_1, LIF_1_1_1_1, LIF_1_3_1, LIF_1_1_3, LIF_1_2_2, LIF_1_1_2_1, LIF_1_1_1_1_1, LIF_1_4, 4LIF, ANN)')
args = parser.parse_args()

# 兼容新旧 gym / gymnasium
def env_reset(env):
    out = env.reset(seed=args.seed)
    return out[0] if isinstance(out, tuple) else out

def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, _ = out
        return obs, reward, terminated or truncated
    return out

def train(args):
    # Environment
    env = []
    env = gym.make(args.env_name)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    args.hidden_size = 32
    from models.model_pre import GaussianPolicy
    agent_pre = SAC(env.observation_space.shape[0], env.action_space, args, GaussianPolicy)
    agent_pre.policy.requires_grad = False
    agent_pre.critic.requires_grad = False
    args.hidden_size = 128
    if(args.model_name == "LIF"):
        from models.model_lif import GaussianPolicy
    elif(args.model_name == "HH"):
        from models.model_hh import GaussianPolicy
    elif(args.model_name == "LIF_HH"):
        from models.model_lif_hh import GaussianPolicy
    elif(args.model_name == "LIF_1_3"):
        from models.models_4LIF.model_lif_1_3 import GaussianPolicy
    elif(args.model_name == "LIF_2_2"):
        from models.models_4LIF.model_lif_2_2 import GaussianPolicy
    elif(args.model_name == "LIF_1_2_1"):
        from models.models_4LIF.model_lif_1_2_1 import GaussianPolicy
    elif(args.model_name == "LIF_1_1_1_1"):
        from models.models_4LIF.model_lif_1_1_1_1 import GaussianPolicy
    elif(args.model_name == "LIF_1_3_1"):
        from models.models_5LIF.model_lif_1_3_1 import GaussianPolicy
    elif(args.model_name == "LIF_1_1_3"):
        from models.models_5LIF.model_lif_1_1_3 import GaussianPolicy
    elif(args.model_name == "LIF_1_2_2"):
        from models.models_5LIF.model_lif_1_2_2 import GaussianPolicy
    elif(args.model_name == "LIF_1_1_2_1"):
        from models.models_5LIF.model_lif_1_1_2_1 import GaussianPolicy
    elif(args.model_name == "LIF_1_1_1_1_1"):
        from models.models_5LIF.model_lif_1_1_1_1_1 import GaussianPolicy
    elif(args.model_name == "LIF_1_4"):
        from models.models_5LIF.model_lif_1_4 import GaussianPolicy
    elif(args.model_name == "4LIF"):
        from models.model_lif import GaussianPolicy
        args.hidden_size = 512
    elif(args.model_name == "ANN"):
        from models.model_ann import GaussianPolicy
    agent = []
    agent = SAC(env.observation_space.shape[0], env.action_space, args, GaussianPolicy)

    # Memory
    memory = []
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    #reward
    negative_reward = -10.0
    x_bound = 1.0

    wins = 5

    agent_pre.load_model(path='sac_pre.pth')
    test = args.eval

    reward_dict_iteration = []

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env_reset(env)
        state_dict = []
        action_dict = []
        reward_dict = []
        mask_dict = []
        next_state_dict = []
        
        while not done:
            steps = len(state_dict)
            if(steps < wins):
                action = agent_pre.select_action(state)  # Sample action from policy
            else:
                if args.start_steps > total_numsteps:
                    action = env.action_space.sample()  # Sample random action
                else:
                    state_tmp = torch.stack([torch.zeros_like(state_dict[0].view(-1))]*wins, dim=0)
                    for i in range(wins):
                        state_tmp[i,:,...] = next_state_dict[steps-wins+i]
                    action = agent.select_action(state_tmp)

            if len(memory) > args.batch_size and not test:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    updates += 1

            next_state, reward, done = env_step(env, action) # Step
            x, x_dot, theta, theta_dot = next_state
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # x_threshold 4.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            if (abs(x) > x_bound):
                r1 = 0.5 * negative_reward
            else:
                r1 = negative_reward * abs(x) / x_bound + 0.5 * (-negative_reward)
            if (abs(theta) > 0.418):
                r2 = 0.5 * negative_reward
            else:
                r2 = negative_reward * abs(theta) / 0.418 + 0.5 * (-negative_reward)
            reward = r1 + r2
            if done:
                reward += negative_reward
            if(steps < wins):
                total_numsteps -= 1
            else:
                reward_dict_iteration.append(reward)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            
            state_dict.append(torch.FloatTensor(state))
            action_dict.append(torch.tensor(action, dtype=torch.float32))
            reward_dict.append(torch.tensor(reward, dtype=torch.float32).view(1,-1))
            next_state_dict.append(torch.tensor(next_state, dtype=torch.float32))
            mask_dict.append(torch.tensor(mask, dtype=torch.float32).view(1,-1))
            
            steps = len(state_dict)
            if(steps < wins):
                pass
            else:
                state_dict_tmp = torch.stack([torch.zeros_like(state_dict[0])]*wins, dim=0)
                action_dict_tmp = torch.stack([torch.zeros_like(action_dict[0])]*wins, dim=0)
                reward_dict_tmp = torch.stack([torch.zeros_like(reward_dict[0])]*wins, dim=1)
                next_state_dict_tmp = torch.stack([torch.zeros_like(next_state_dict[0])]*wins, dim=0)
                mask_dict_tmp = torch.stack([torch.zeros_like(mask_dict[0])]*wins, dim=1)
                for i in range(wins):
                    state_dict_tmp[i,:,...] = state_dict[steps-wins+i]
                    action_dict_tmp[i,:,...] = action_dict[steps-wins+i]
                    reward_dict_tmp[:,i,...] = reward_dict[steps-wins+i]
                    next_state_dict_tmp[i,:,...] = next_state_dict[steps-wins+i]
                    mask_dict_tmp[:,i,...] = mask_dict[steps-wins+i]
                memory.push(state_dict_tmp,action_dict_tmp,reward_dict_tmp,next_state_dict_tmp,mask_dict_tmp)
                
            state = next_state
            
            if total_numsteps >= args.num_steps:
                break

        if total_numsteps >= args.num_steps:
            # Route topologies into 4-LIF / 5-LIF subdirs
            _4LIF = {"LIF_1_3", "LIF_2_2", "LIF_1_2_1", "LIF_1_1_1_1", "LIF_ring"}
            _5LIF = {"LIF_1_3_1", "LIF_1_1_3", "LIF_1_2_2", "LIF_1_1_2_1", "LIF_1_1_1_1_1", "LIF_1_4"}
            if args.model_name in _4LIF:
                subdir = "4-LIF"
            elif args.model_name in _5LIF:
                subdir = "5-LIF"
            else:
                subdir = args.model_name
            output_dir = "./record/{}".format(subdir)
            os.makedirs(output_dir, exist_ok=True)
            np.save("{}/reward_iteration_{}_seed{}.npy".format(
                        output_dir, args.model_name, args.seed),
                    {"reward_dict":np.array(reward_dict_iteration),
                    "iteration":np.array(total_numsteps),
                    "model_name":args.model_name})
            break

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if test and i_episode%10 == 0:
            avg_reward = 0.
            avg_step = 0.
            episodes = 10
            for _  in range(episodes):
                state = env_reset(env)
                episode_reward = 0
                episode_step = 0
                done = False
                state_dict_test = []
                frames = []
                while not done:
                    state_dict_test.append(torch.FloatTensor(state).view(-1))
                    steps = len(state_dict_test)
                    if(steps < wins):
                        state_tmp = torch.stack([torch.FloatTensor(state).view(-1)]*wins, dim=0)
                        action = agent_pre.select_action(state_tmp, evaluate=True)  # Sample action from policy
                    else:
                        state_tmp = torch.stack([torch.zeros_like(state_dict_test[0])]*wins, dim=0)
                        for i in range(wins):
                            state_tmp[i,:,...] = state_dict_test[steps-wins+i]
                        action = agent.select_action(state_tmp, evaluate=True)  # Sample action from policy
                    next_state, reward, done = env_step(env, action)
                    
                    episode_reward += reward
                    episode_step += 1

                    state = next_state
                avg_reward += episode_reward
                avg_step += episode_step
            avg_reward /= episodes
            avg_step /= episodes

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()


train(args)