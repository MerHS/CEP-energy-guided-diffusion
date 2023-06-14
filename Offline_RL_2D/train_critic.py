from copy import deepcopy
import os
import cv2
# import gym
# import d4rl
# import scipy
import tqdm
import functools

import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
# from utils import get_args, pallaral_eval_policy
from utils import get_args, simple_eval_policy
# from dataset.dataset import D4RL_dataset
from dataset.orcaset import ORCADataset
from vae import VanillaVAE

LOAD_FAKE=False
WIDTH = 10
HEIGHT = 10
RADIUS = 0.1
VIS_RANGE = 2

def train_critic(args, score_model, data_loader, start_epoch=0):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 100
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 2
    save_interval = 10

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}
            if (epoch < 50):
                score_model.q[0].update_q0(data)
            loss2 = score_model.q[0].update_qt(data)
            avg_loss += loss2
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            for guidance_scale in args.s:
                score_model.q[0].guidance_scale = guidance_scale

                mean, std = args.eval_func(score_model.select_actions)
                # mean = np.mean([envs[i].buffer_return for i in range(args.seed_per_evaluation)])
                # std = np.std([envs[i].buffer_return for i in range(args.seed_per_evaluation)])

                args.writer.add_scalar("eval/rew{}".format(guidance_scale), mean, global_step=epoch)
                args.writer.add_scalar("eval/std{}".format(guidance_scale), std, global_step=epoch)
            score_model.q[0].guidance_scale = 1.0

        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.q[0].state_dict(), os.path.join("./models_rl", str(args.expid), "critic_ckpt{}.pth".format(epoch+1)))
        args.writer.add_scalar("critic/loss", avg_loss / num_items, global_step=epoch)
        data_p = [0, 10, 25, 50, 75, 90, 100]
        if args.debug:
            args.writer.add_scalars("target/mean", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_mean, data_p))}, global_step=epoch)
            args.writer.add_scalars("target/std", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].all_std, data_p))}, global_step=epoch)
            args.writer.add_scalars("target/debug", {str(p): d for p, d in zip(data_p, np.percentile(score_model.q[0].debug_used, data_p))}, global_step=epoch)

def gauss(len, mu, sigma):
    x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

gauss_200 = gauss(200, 0, 0.05)
gauss_100 = gauss_200[50:150, 50:150]

class TestEnv():
    def __init__(self, ckpt_path, device, eval_episodes=20):
        val_file = '/home/gpuadmin/work/CEP-energy-guided-diffusion/Offline_RL_2D/data/val_fake.pkl'
        val_obs = torch.load(val_file)

        ckpt = torch.load(ckpt_path)
        self.model = VanillaVAE(1, 24)
        self.model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()})
        self.model.eval()
        self.model.to(device=device)
        self.device = device

        self.start_episodes = []
        self.episodes = []
        self.reward_max = []

        for path_target in val_obs:
            episode = []
            for path, target in zip(path_target[0], path_target[1]):
                pos, latent, vel, reward, next_pos, done = path[0]
                pos = np.array(pos)
                target = np.array(target)

                self.reward_max.append(np.linalg.norm(pos - target) * 0.3)
                episode.append([[pos, target, np.array(latent)], False])
            self.episodes.append(episode)
        
        self.start_episodes = deepcopy(self.episodes)
        self.len_agents = sum([len(episode) for episode in self.episodes])
        self.rewards = [0 for _ in range(self.len_agents)]

    def get_states(self):
        states = []
        for episode in self.episodes:
            for agent, done in episode:
                if not done:
                    states.append(np.concatenate(agent))
        
        return states

    def reset(self):
        self.episodes = deepcopy(self.start_episodes)
        self.rewards = [0 for _ in range(self.len_agents)]

        return self.get_states()

    def step(self, actions):
        agents = []
        for episode in self.episodes:
            for agent, done in episode:
                if not done:
                    agents.append((agent, agent[0].copy()))

        for i, (agent, _) in enumerate(agents):
            agent[0] = agent[0] + actions[i] * (1/30)
        
        total_reward = 0
        agent_i = 0
        all_done = True
        epi_i = 0
        for episode in self.episodes:
            plane_w = int((WIDTH + 2 * VIS_RANGE) / VIS_RANGE * 100)
            plane_h = int((HEIGHT + 2 * VIS_RANGE) / VIS_RANGE * 100)
            plane = np.zeros((plane_w, plane_h))

            pos_list = [agent[0] for agent, _ in episode]
            planes = [plane.copy() for _ in range(episode)]

            for i, plane in enumerate(planes):
                if episode[i][1]:
                    continue
                for j, pos in enumerate(old_pos):
                    if i == j:
                        continue
                    pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
                    pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)

                    pos_x = int(np.clip(pos_x, 100, plane_w - 100))
                    pos_y = int(np.clip(pos_y, 100, plane_h - 100))
                    # print(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50].shape, pos, pos_x, pos_y)
                    plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50] = np.maximum(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50], gauss_100)

            for pos in pos_list:
                pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
                pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)

                pos_x = int(np.clip(pos_x, 100, plane_w - 100))
                pos_y = int(np.clip(pos_y, 100, plane_h - 100))
                # print(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50].shape, pos, pos_x, pos_y)
                plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50] = np.maximum(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50], gauss_100)
        
            for i, agent_done in enumerate(episode):
                agent, done = agent_done
                pos, target, _ = agent
                if done:
                    continue


                old_pos = agents[agent_i][1]
                dist = np.linalg.norm(pos - target)
                dist_diff = np.linalg.norm(old_pos - target) - dist
                reward = dist_diff * 0.3

                # if agent_i == 3:
                #     print(dist_diff, reward)

                if dist < 0.05:
                    agent_done[1] = True

                coll_reward = 0
                for j in range(len(episode)):
                    if i == j:
                        continue
                    if np.linalg.norm(pos - episode[j][0][0]) < 2 * RADIUS:
                        coll_reward = -1 # collision
                        break

                reward += coll_reward

                pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
                pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)
                pos_x = int(np.clip(pos_x, 100, plane_w - 100))
                pos_y = int(np.clip(pos_y, 100, plane_h - 100))
                view = plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50]

                view = cv2.resize(view, (64, 64))
                view_tensor = torch.from_numpy(view).unsqueeze(0).unsqueeze(0).float().to(device=self.device)
                with torch.no_grad():
                    latent = self.model.reparameterize(*self.model.encode(view_tensor)).squeeze().cpu().detach().numpy()

                agent[2] = latent

                agent_i += 1
                all_done = all_done and agent_done[1]

                self.rewards[epi_i + i] += reward
                total_reward += reward

            epi_i += len(episode)

        return self.get_states(), total_reward, all_done
                

def critic(args):
    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))
    writer = SummaryWriter("./logs/" + str(args.expid))
    
    # env = gym.make(args.env)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vae_path = "/home/gpuadmin/work/CEP-energy-guided-diffusion/Offline_RL_2D/vae.ckpt"
    
    test_env = TestEnv(vae_path, device=args.device, eval_episodes=args.seed_per_evaluation)
    # test_act = np.random.rand(1010, 2)
    # test_env.reset()
    # test_env.step(test_act)

    args.eval_func = functools.partial(simple_eval_policy, env=test_env, seed=args.seed, eval_episodes=args.seed_per_evaluation, diffusion_steps=args.diffusion_steps)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    state_dim = 24 + 2 + 2
    action_dim = 2
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.q[0].to(args.device)

    print("loading actor...")
    ckpt = torch.load(args.actor_load_path, map_location=args.device)[0]
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    score_model.load_state_dict(ckpt)
    
    # dataset = D4RL_dataset(args)
    batch_size = 256
    dataset = ORCADataset(args)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    score_model.q[0].guidance_scale = 0.0

    # mean, std = args.eval_func(score_model.select_actions)
    
    # generate support action set
    if os.path.exists(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_actions = torch.Tensor(np.load(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to(args.device)
    else:
        allstates = dataset.states[:].cpu().numpy()
        actions = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // batch_size + 1)):
            fa = score_model.sample(states, sample_per_state=args.M, diffusion_steps=args.diffusion_steps)
            actions.append(torch.tensor(fa, device=args.device))
        # dataset.fake_actions = torch.tensor(np.array(actions, dtype=np.float32), device=args.device)
        dataset.fake_actions = torch.cat(actions, dim=0)
        if LOAD_FAKE:
            np.save(args.actor_load_path+ "actions{}_raw.npy".format(args.diffusion_steps), actions)

    # fake next action
    if os.path.exists(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps)) and LOAD_FAKE:
        dataset.fake_next_actions = torch.Tensor(np.load(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps)).astype(np.float32)).to(args.device)
    else:
        allstates = dataset.next_states[:].cpu().numpy()
        actions = []
        for states in tqdm.tqdm(np.array_split(allstates, allstates.shape[0] // batch_size + 1)):
            fa = score_model.sample(states, sample_per_state=args.M, diffusion_steps=args.diffusion_steps)
            actions.append(torch.tensor(fa, device=args.device))
        # dataset.fake_next_actions = torch.tensor(np.array(actions, dtype=np.float32), device=args.device)
        dataset.fake_next_actions = torch.cat(actions, dim=0)
        if LOAD_FAKE:
            np.save(args.actor_load_path+ "next_actions{}_raw.npy".format(args.diffusion_steps), actions)

    print("training critic")
    train_critic(args, score_model, data_loader, start_epoch=0)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    if "antmaze" not in args.env:
        args.M = 16
    else:
        args.M = 32
    if args.s is None:
        args.s = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0] if "antmaze" in args.env else [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    critic(args)