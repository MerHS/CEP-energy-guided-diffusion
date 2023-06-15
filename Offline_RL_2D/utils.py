import argparse
# import gym
import time
import numpy as np
import torch
# from tensorboard.backend.event_processing import event_accumulator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-expert-v2") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cpu", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--n_behavior_epochs', type=int, default=600)
    parser.add_argument('--actor_load_path', type=str, default="/home/gpuadmin/work/CEP-energy-guided-diffusion/Offline_RL_2D/models_rl/default/behavior_ckpt500.pth")
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--M', type=int, default=16)               # support action number
    parser.add_argument('--seed_per_evaluation', type=int, default=10)
    parser.add_argument('--s', type=float, nargs="*", default=None)# guidance scale
    parser.add_argument('--method', type=str, default="CEP")
    parser.add_argument('--q_alpha', type=float, default=None)     
    print("**************************")
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch =1
        args.critic_epoch =1
        args.env = "antmaze-medium-play-v2"
    if args.q_alpha is None:
        args.q_alpha = args.alpha
    print(args)
    return args

def bandit_get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="8gaussians") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)             # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--expid", default="default", type=str)    # 
    parser.add_argument("--device", default="cuda", type=str)      #
    parser.add_argument("--save_model", default=1, type=int)       #
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=3.0)        # beta parameter in the paper, use alpha because of legacy
    parser.add_argument('--diffusion_steps', type=int, default=15)
    parser.add_argument('--method', type=str, default="CEP")
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args

# def pallaral_eval_policy(policy_fn, env_name, seed, eval_episodes=20, diffusion_steps=15):
#     eval_envs = []
#     for i in range(eval_episodes):
#         env = gym.make(env_name)
#         eval_envs.append(env)
#         env.seed(seed + 1001 + i)
#         env.buffer_state = env.reset()
#         env.buffer_return = 0.0

#     ori_eval_envs = [env for env in eval_envs]
    
#     t = time.time()
#     while len(eval_envs) > 0:
#         new_eval_envs = []
#         states = np.stack([env.buffer_state for env in eval_envs])
#         actions = policy_fn(states, diffusion_steps=diffusion_steps)
#         for i, env in enumerate(eval_envs):
#             state, reward, done, info = env.step(actions[i])
#             env.buffer_return += reward
#             env.buffer_state = state
#             if not done:
#                 new_eval_envs.append(env)
#         eval_envs = new_eval_envs

#     print(time.time() - t)
#     mean = np.mean([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
#     std = np.std([ori_eval_envs[i].buffer_return for i in range(eval_episodes)])
#     print("reward {} +- {}".format(mean, std))

#     return ori_eval_envs

def simple_eval_policy(policy_fn, env, seed, eval_episodes=20, diffusion_steps=15):
    # env = gym.make(env_name)
    # env.seed(seed+561)
    all_rewards = []

    # for _ in range(eval_episodes):
    obs = env.reset()
    # total_reward = 0.
    done = False
    i = 0
    while not done and i < 10: # 10 sec timeout
        with torch.no_grad():
            obs = np.array(obs)
            # obs = torch.tensor(, dtype=torch.float32).cuda()
            actions = policy_fn(obs)
            # actions = [a.cpu().numpy() for a in actions]
        next_obs, _, done = env.step(actions)
        # total_reward += reward
        if done:
            break
        else:
            obs = next_obs
        i += 1
    
    all_rewards = env.rewards
    max_rewards = env.reward_max
    norm_rewards = [r / (m + 1e-6) for r, m in zip(all_rewards, max_rewards)]

    return np.mean(norm_rewards), np.std(norm_rewards)