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
    def save_checkpoint(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),

            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),

            "log_alpha": self.log_alpha.detach().cpu(),

            "replay_buffer": self.replay_buffer.serialize(),

            "total_steps": self.total_steps,
            "episode": self.episode,
        }

        torch.save(checkpoint, path)
        print(f"[✓] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu")

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

        self.log_alpha.data.copy_(checkpoint["log_alpha"])

        self.replay_buffer.deserialize(checkpoint["replay_buffer"])

        self.total_steps = checkpoint["total_steps"]
        self.episode = checkpoint["episode"]

        print(f"[✓] Loaded checkpoint from {path}")


