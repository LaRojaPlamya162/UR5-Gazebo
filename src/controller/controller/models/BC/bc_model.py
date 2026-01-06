import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributions as D

LOG_STD_MIN = -5
LOG_STD_MAX = 2
class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Level 1: output parameters of a distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Return a distribution π(a|s)
        """
        h = self.backbone(state)
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return D.Normal(mean, std)

    def act(self, state, deterministic=False):
        """
        Sample or take mean action (for deployment)
        """
        dist = self.forward(state)
        if deterministic:
            return dist.mean
        else:
            return dist.sample()

def cut_back_attribute(example, col_name, n):
        example[col_name] = example[col_name][:n]
        return example 
def bc_nll_loss(dist, actions):
    log_prob = dist.log_prob(actions)
    return -log_prob.sum(dim=-1).mean()
if __name__ == '__main__':
    dataset = load_dataset("lerobot/berkeley_autolab_ur5", split = "train")
    dataset = dataset.with_format("torch")
    dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "action", "n" : 6})
    dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "observation.state", "n" : 6})
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 256,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        drop_last = True
    )
    model = BCPolicy(state_dim = len(dataset[0]['observation.state']),
                     action_dim = len(dataset[0]['action']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
        #for i in range(len(dataset)):
            state = batch['observation.state']
            action = batch['action']

            dist = model(state)
            loss = bc_nll_loss(dist, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(dataloader):.6f}")
    
    print("Training finised")
    torch.save(model.state_dict(), 'src/controller/controller/models/BC/bc_model_v2.pth')
    #torch.save(model.state_dict(), 'bc_model.pth')
    print("Model loaded!")