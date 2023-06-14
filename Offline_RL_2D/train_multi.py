import os
# import gym
# import d4rl
# import scipy
import tqdm
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusion_SDE.loss import loss_fn
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet
from utils import get_args
from dataset.orcaset import ORCADataset

def train_behavior(args, score_model, data_loader, start_epoch=0, rank=0):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 500
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    save_interval = 30
    
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(1000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}

            s = data['s']
            a = data['a']
            score_model.module.condition = s
            loss = loss_fn(score_model, a, args.marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            score_model.module.condition = None

            avg_loss += loss
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            
        if rank == 0 and ((epoch % save_interval == (save_interval - 1)) or epoch == n_epochs-1):
            save_path = os.path.join("./models_rl", str(args.expid), "behavior_ckpt{}.pth".format(epoch+1))
            torch.save([score_model.state_dict(), optimizer.state_dict()], save_path)
            print(f"saved to {save_path}")
        if args.writer:
            args.writer.add_scalar("actor/loss", avg_loss / num_items, global_step=epoch)

def behavior(rank, args, world_size):
    # The diffusion behavior training pipeline is copied directly from https://github.com/ChenDRAG/SfBC/blob/master/train_behavior.py
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    args.device = torch.device("cuda:{}".format(rank))

    for dir in ["./models_rl", "./logs"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.expid))):
        os.makedirs(os.path.join("./models_rl", str(args.expid)))

    if rank == 0:
        writer = SummaryWriter("./logs/" + str(args.expid))
    else:
        writer = None
    
    # env = gym.make(args.env)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    state_dim = 24 + 2 + 2
    action_dim = 2
    # max_action = float(env.action_space.high[0])
    args.writer = writer
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)

    # score_model = torch.compile(score_model)
    score_model = DDP(score_model, device_ids=[rank])
    
    # dataset = D4RL_dataset(args)
    dataset = ORCADataset(args)
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    print("training behavior")
    train_behavior(args, score_model, data_loader, start_epoch=0, rank=rank)
    print("finished")

def main(args):
    world_size = 2
    mp.spawn(behavior,
        args=(args, world_size,),
        nprocs=world_size,
        join=True)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    args = get_args()
    main(args)