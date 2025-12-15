from controller.models.SAC.network import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.q1 = Critic(state_dim, action_dim)
        self.q2 = Critic(state_dim, action_dim)

        self.q1_target = Critic(state_dim, action_dim)
        self.q2_target = Critic(state_dim, action_dim)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005

    def update(self, replay, batch_size=256):
        s, a, r, s_, d = replay.sample(batch_size)

        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        with torch.no_grad():
            a_, logp_ = self.actor.sample(s_)
            q1_t = self.q1_target(s_, a_)
            q2_t = self.q2_target(s_, a_)
            q_target = r + self.gamma * (1 - d) * (torch.min(q1_t, q2_t) - self.alpha * logp_)

        # Critic update
        q1_loss = F.mse_loss(self.q1(s, a), q_target)
        q2_loss = F.mse_loss(self.q2(s, a), q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # Actor update
        a_new, logp = self.actor.sample(s)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        actor_loss = (self.alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target
        for target, source in zip(self.q1_target.parameters(), self.q1.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

        for target, source in zip(self.q2_target.parameters(), self.q2.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
