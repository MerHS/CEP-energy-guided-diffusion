
import torch
import numpy as np
import torch.nn.functional as F

# Dataset iterator
class ORCADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args=args

        train_file = '/home/gpuadmin/work/CEP-energy-guided-diffusion/Offline_RL_2D/data/train/train.pkl'
        # train_file = '/home/gpuadmin/work/CEP-energy-guided-diffusion/Offline_RL_2D/data/test/val.pkl'
        train_obs = torch.load(train_file)
        total_len = 0
        for episode in train_obs:
            for path in episode[0]:
                total_len += len(path)
        
        states = []
        actions = []
        next_states = []
        rewards = []
        is_finished = []

        for episode in train_obs:
            for path, target in zip(episode[0], episode[1]):
                target = np.array(target)
                last_latent = None
                last_pos = None
                last_state = None
                for pos, latent, vel, reward, next_pos, done in path:
                    state = np.concatenate((pos, target, latent))
                    states.append(state)
                    actions.append(vel)
                    rewards.append(reward)
                    is_finished.append(False)

                    if last_state is not None:
                        next_states.append(state)

                    last_latent = latent
                    last_pos = next_pos
                    last_state = state

                if last_latent is not None:
                    next_states.append(np.concatenate((last_pos, target, last_latent)))

                is_finished[-1] = True

        self.device = args.device
        self.states = torch.from_numpy(np.array(states)).float().to(self.device)
        self.actions = torch.from_numpy(np.array(actions)).float().to(self.device)
        self.next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        reward = torch.from_numpy(np.array(rewards)).view(-1, 1).float().to(self.device)
        self.is_finished = torch.from_numpy(np.array(is_finished)).view(-1, 1).float().to(self.device)

        self.rewards = reward
        print("dql dataloard loaded")
        
        self.len = self.states.shape[0]
        print(self.len, "data loaded")

    def __getitem__(self, index):
        data = {'s': self.states[index % self.len],
                'a': self.actions[index % self.len],
                'r': self.rewards[index % self.len],
                's_':self.next_states[index % self.len],
                'd': self.is_finished[index % self.len],
                'fake_a':self.fake_actions[index % self.len] if hasattr(self, "fake_actions") else 0.0,  # self.fake_actions <D, 16, A>
                'fake_a_':self.fake_next_actions[index % self.len] if hasattr(self, "fake_next_actions") else 0.0,  # self.fake_next_actions <D, 16, A>
            }
        return data

    def __add__(self, other):
        pass
    def __len__(self):
        return self.len