import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from controller.models.IQL.network import QNetwork, VNetwork, Policy

class IQLAgent:
    def __init__(self, state_dim, action_dim,
                 tau=0.7, beta=3.0, gamma=0.99):
        self.q = QNetwork(state_dim, action_dim)
        self.v = VNetwork(state_dim)
        self.policy = Policy(state_dim, action_dim)

        self.q_opt = torch.optim.Adam(self.q.parameters(), 3e-4)
        self.v_opt = torch.optim.Adam(self.v.parameters(), 3e-4)
        self.pi_opt = torch.optim.Adam(self.policy.parameters(), 3e-4)

        self.tau = tau
        self.beta = beta
        self.gamma = gamma  
    def update_v(self, s, a):
        q = self.q(s, a).detach()
        v = self.v(s)

        diff = q - v
        weight = torch.where(diff > 0, self.tau, 1 - self.tau)
        loss = (weight * diff.pow(2)).mean()

        self.v_opt.zero_grad()
        loss.backward()
        self.v_opt.step()
    def update_q(self, s, a, r, s_next, done):
        with torch.no_grad():
            target = r + self.gamma * (1 - done) * self.v(s_next)

        loss = F.mse_loss(self.q(s, a), target)

        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
    def update_policy(self, s, a):
        with torch.no_grad():
            adv = self.q(s, a) - self.v(s)
            weight = torch.exp(adv / self.beta).clamp(max=100)

        mean, log_std = self.policy(s)
        dist = torch.distributions.Normal(mean, log_std.exp())
        log_prob = dist.log_prob(a).sum(-1)

        loss = -(weight * log_prob).mean()

        self.pi_opt.zero_grad()
        loss.backward()
        self.pi_opt.step()
    def train_step(self, batch):
        s, a, r, s_next, done = batch
        self.update_v(s, a)
        self.update_q(s, a, r, s_next, done)
        self.update_policy(s, a)
    def save(self, path):
        torch.save({
            "q": self.q.state_dict(),
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.q.load_state_dict(ckpt["q"])
        self.v.load_state_dict(ckpt["v"])
        self.policy.load_state_dict(ckpt["policy"])
