import torch
import torch.nn.functional as F
from .network import Actor, Critic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACAgent:
    def __init__(self, state_dim, action_dim, action_space=None):
        self.actor = Actor(state_dim, action_dim, action_space).to(DEVICE)

        self.q1 = Critic(state_dim, action_dim).to(DEVICE)
        self.q2 = Critic(state_dim, action_dim).to(DEVICE)
        self.q1_target = Critic(state_dim, action_dim).to(DEVICE)
        self.q2_target = Critic(state_dim, action_dim).to(DEVICE)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)

        # Entropy
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005

    # ================= ACTION =================
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, mean = self.actor.sample(state)

        return mean.cpu().numpy().flatten() if deterministic else action.cpu().numpy().flatten()

    # ================= UPDATE =================
    def update(self, replay, batch_size=256):
        if replay.size < batch_size:
            return

        s, a, r, s_, d = replay.sample(batch_size)

        alpha = self.log_alpha.exp().clamp(1e-4, 10)

        # -------- Critic --------
        with torch.no_grad():
            a_next, logp_next, _ = self.actor.sample(s_)
            q1_t = self.q1_target(s_, a_next)
            q2_t = self.q2_target(s_, a_next)

            min_q = torch.min(q1_t, q2_t)
            target_q = r + self.gamma * (1 - d) * (min_q - alpha * logp_next)

        q1_loss = F.mse_loss(self.q1(s, a), target_q)
        q2_loss = F.mse_loss(self.q2(s, a), target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_opt.step()

        # -------- Actor --------
        a_new, logp, _ = self.actor.sample(s)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))

        actor_loss = (alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # -------- Alpha --------
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # -------- Soft update --------
        for t, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

        for t, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
