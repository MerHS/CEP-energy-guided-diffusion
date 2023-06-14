import random
import functools
from copy import deepcopy

import numpy as np
import torch
import cv2
import itertools
import pygame

from utils import get_args
from vae import VanillaVAE
from diffusion_SDE.model import ScoreNet
from diffusion_SDE.schedule import marginal_prob_std

VAL_SIZE = 100
TRAIN_SIZE = 3000

WIDTH = 10
HEIGHT = 10
RADIUS = 0.18
VIS_RANGE = 2
TIMESTEP = 1/30.

colors = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 120, 0),
    (255, 120, 120),
]

def gauss(len, mu, sigma):
    x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

gauss_200 = gauss(200, 0, 0.05)
gauss_100 = gauss_200[50:150, 50:150]

class TestEnv():
    def __init__(self, vae_path, n_agents, device):
        vae_weight = torch.load(vae_path, map_location='cpu')
        self.vae_model = VanillaVAE(1, 24)
        self.vae_model.load_state_dict({k.replace('model.', ''): v for k, v in vae_weight['state_dict'].items()})
        self.vae_model.eval()
        self.vae_model.to(device=device)

        self.device = device
        self.n_agents = n_agents

        self.reset()

    def gen_latent(self):
        latent_all = []

        plane_w = int((WIDTH + 2 * VIS_RANGE) / VIS_RANGE * 100)
        plane_h = int((HEIGHT + 2 * VIS_RANGE) / VIS_RANGE * 100)
        plane = np.zeros((plane_w, plane_h))

        planes = [plane.copy() for _ in range(self.n_agents)]

        for i, plane in enumerate(planes):
            for j, pos in enumerate(self.pos):
                # if i == j:
                #     continue
                pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
                pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)

                pos_x = int(np.clip(pos_x, 100, plane_w - 100))
                pos_y = int(np.clip(pos_y, 100, plane_h - 100))
                # print(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50].shape, pos, pos_x, pos_y)
                plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50] = np.maximum(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50], gauss_100)

        for pos in self.pos:
            pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
            pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)

            pos_x = int(np.clip(pos_x, 100, plane_w - 100))
            pos_y = int(np.clip(pos_y, 100, plane_h - 100))
            # print(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50].shape, pos, pos_x, pos_y)
            plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50] = np.maximum(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50], gauss_100)
    
        for i, pos in enumerate(self.pos):
            pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
            pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)
            pos_x = int(np.clip(pos_x, 100, plane_w - 100))
            pos_y = int(np.clip(pos_y, 100, plane_h - 100))
            view = plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50]

            view = cv2.resize(view, (64, 64))
            view_tensor = torch.from_numpy(view).unsqueeze(0).unsqueeze(0).float().to(device=self.device)
            with torch.no_grad():
                latent = self.vae_model.reparameterize(*self.vae_model.encode(view_tensor)).squeeze().cpu().detach().numpy()

            latent_all.append(latent)
        
        self.latent = latent_all

    def get_states(self):
        states = []

        for pos, target, latent, done in zip(self.pos, self.target, self.latent, self.done):
            if not done:
                states.append(np.concatenate([pos, target, latent]))
        
        return states

    def reset(self):
        self.pos = []
        self.vel = []
        self.target = []
        self.done = []
        self.latent = []

        for i in range(self.n_agents):
            target_done = True
            while target_done:
                target_done = False

                pos = np.array([random.uniform(1, WIDTH - 1), random.uniform(1, HEIGHT - 1)])
                target = np.array([WIDTH - pos[0] + random.uniform(-1, 1), HEIGHT - pos[1] + random.uniform(-1, 1)])

                for curr_pos in self.pos:
                    if np.linalg.norm(pos - curr_pos) < 2 * RADIUS:
                        target_done = True
                        break
                
                for curr_target in self.target:
                    if np.linalg.norm(target - curr_target) < 2 * RADIUS:
                        target_done = True
                        break

            vel = target - pos
            vel = vel / (np.linalg.norm(vel) + 1e-6)

            self.pos.append(pos)
            self.vel.append(vel)
            self.target.append(target)
            self.done.append(False)

        self.gen_latent()

        self.rewards = [0 for _ in range(self.n_agents)]

    def step(self, actions):
        orig_pos_list = []
    
        for i, (pos, done) in enumerate(zip(self.pos, self.done)):
            if not done:
                orig_pos_list.append((pos.copy(), i))

        for i, (_, orig_i) in enumerate(orig_pos_list):
            self.vel[orig_i] = actions[i]
            self.pos[orig_i] += self.vel[orig_i] * (1/30.)
        
        total_reward = 0
        all_done = True

        for i, (old_pos, orig_i) in enumerate(orig_pos_list):
            pos = self.pos[orig_i]
            target = self.target[orig_i]
            
            dist = np.linalg.norm(pos - target)
            dist_diff = np.linalg.norm(old_pos - target) - dist
            reward = dist_diff * 0.3

            if dist < 0.05:
                self.done[orig_i] = True

            coll_reward = 0
            for j in range(self.n_agents):
                if orig_i == j:
                    continue
                if np.linalg.norm(pos - self.pos[j]) < 0.1:
                    coll_reward = -1 # collision
                    break

            reward += coll_reward

            all_done = all_done and self.done[orig_i]
            self.rewards[orig_i] += reward
            total_reward += reward
        
        self.gen_latent()

        return self.get_states(), total_reward, all_done
    

def main(args):
    # TODO:
    actor_path = '/Users/kinetc/work/CEP-energy-guided-diffusion/Offline_RL_2D/behavior_ckpt500.pth'
    vae_path = '/Users/kinetc/work/CEP-energy-guided-diffusion/Offline_RL_2D/last.ckpt'

    actor_weight = torch.load(actor_path, map_location='cpu')[0]
    actor_weight = {k.replace("module.", ""): v for k, v in actor_weight.items()}
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)

    score_model = ScoreNet(input_dim=2 + 2 + 24 + 2, output_dim=2, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.load_state_dict(actor_weight)
    score_model.q[0].guidance_scale = 0.0

    env = TestEnv(vae_path, 10, args.device)
    env.reset()

    pygame.init()
    dim = (640, 640)
    # origin = np.array(dim) / 2
    origin = np.array([0, 0])
    scale = 6

    screen = pygame.display.set_mode(dim)

    clock = pygame.time.Clock()

    def draw_agent(pos, radius, color):
        pygame.draw.circle(
            screen, 
            color, 
            np.rint(pos * scale + origin).astype(int), 
            int(round(radius * scale)), 0)

    def draw_velocity(pos, vel):
        pygame.draw.line(
            screen, 
            pygame.Color(0, 255, 255), 
            np.rint(pos * scale + origin).astype(int), 
            np.rint((pos + vel) * scale + origin).astype(int), 1)

    tick = 0
    running = True
    done = False
    while running:
        clock.tick(30)

        tick += 1

        screen.fill(pygame.Color(0, 0, 0))

        if not done: 
            with torch.no_grad():
                obs = np.array(env.get_states())
                actions = score_model.select_actions(obs)

            _, reward, done = env.step(actions)
        
        # with torch.no_grad():
        #     Z = model.decode(torch.tensor(last_latent).unsqueeze(0).float()).reshape(64, 64, 1).numpy()
        #     Z = np.repeat(Z * 255, 3, axis=2)

        # surf = pygame.surfarray.make_surface(Z)
        # screen.blit(surf, (0, 0))

        for pos, vel, target, color in zip(env.pos, env.vel, env.target, itertools.cycle(colors)):
            pos = np.array(pos) * 10
            target = np.array(target) * 10
            vel = np.array(vel) * 10
            radius = RADIUS * 10

            draw_agent(pos, radius, color)
            draw_agent(target, radius / 2, color)
            draw_velocity(pos, vel)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

if __name__ == "__main__":
    args = get_args()
    main(args)