!pip install torch numpy torch_geometric

import sys
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

from torch_geometric.nn import GINConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WaterSortEnv:
    """Discrete Water Sort puzzle environment with completed flags and custom rewards."""
    def __init__(self, num_tubes=5, tube_capacity=4):
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self._build_colors()
        self.reset()

    def _build_colors(self):
        # Colors as integer indices 0..(num_tubes-3)
        self.COLORS = list(range(self.num_tubes - 2))

    def reset(self):
        self._build_colors()
        self.tubes = [[] for _ in range(self.num_tubes)]
        self.moves = 0
        self.max_moves = 50
        self.game_over = False

        pool = self.COLORS * self.tube_capacity
        random.shuffle(pool)
        for i in range(self.num_tubes - 2):
            start = i * self.tube_capacity
            self.tubes[i] = pool[start:start + self.tube_capacity]
        return self._get_state()

    def _get_state(self):
        # shape: (num_tubes, tube_capacity+1)
        state = np.zeros((self.num_tubes, self.tube_capacity + 1), dtype=np.int64)
        for ti, tube in enumerate(self.tubes):
            completed = 1 if tube and all(c == tube[0] for c in tube) else 0
            state[ti, 0] = completed
            for li, color in enumerate(tube):
                state[ti, li + 1] = color + 1
        return state

    def is_valid_move(self, f, t):
        if f == t or not self.tubes[f] or len(self.tubes[t]) >= self.tube_capacity:
            return False
        return (not self.tubes[t]) or (self.tubes[f][-1] == self.tubes[t][-1])

    def step(self, action):
        f, t = action
        self.moves += 1
        reward = -0.01

        if not self.is_valid_move(f, t):
            return self._get_state(), -0.05, self.game_over, {}

        color = self.tubes[f].pop()
        self.tubes[t].append(color)

        if len(self.tubes[t]) == self.tube_capacity and all(c == self.tubes[t][0] for c in self.tubes[t]):
            reward += 1.0

        if self.check_win():
            reward += 10.0
            self.game_over = True

        if self.moves >= self.max_moves and not self.game_over:
            reward -= 5.0
            self.game_over = True

        return self._get_state(), reward, self.game_over, {}

    def check_win(self):
        for tube in self.tubes:
            if not tube:
                continue
            if any(c != tube[0] for c in tube):
                return False
        return True


class WaterSortGNN(nn.Module):
    """GNN+Transformer policy network."""
    def __init__(self, num_tubes, tube_capacity, hidden_dim=128):
        super().__init__()
        vocab = num_tubes + 2
        self.embedding = nn.Embedding(vocab, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(3):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnn_layers.append(GINConv(mlp))

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(hidden_dim, nhead=8), num_layers=2
        )
        self.action_head = nn.Linear(hidden_dim * 2, 1)
        self.num_tubes = num_tubes

    def forward(self, state):
        B, T, L = state.shape
        x = self.embedding(state.to(device))  # (B,T,L,H)
        tube_repr = x.mean(dim=2)             # (B,T,H)
        flat = tube_repr.view(-1, tube_repr.size(-1))
        edge_index = self._build_edge_index(T).to(device)

        for gnn in self.gnn_layers:
            flat = gnn(flat, edge_index)

        tube_repr = flat.view(B, T, -1)
        tube_repr = self.transformer(tube_repr)

        actions = []
        for i in range(T):
            for j in range(T):
                if i == j: continue
                pair = torch.cat([tube_repr[:, i], tube_repr[:, j]], dim=1)
                actions.append(self.action_head(pair))
        return torch.cat(actions, dim=1)

    def _build_edge_index(self, T):
        idx = []
        for i in range(T):
            for j in range(T):
                if i != j:
                    idx.append([i, j])
        return torch.tensor(idx, dtype=torch.long).t()


class PPOTrainer:
    """PPO training with clipped surrogate."""
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def update(self, states, actions, rewards, old_logp):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret = torch.tensor(returns, device=device)
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        logits = self.model(states)
        dist = Categorical(logits=logits.squeeze(-1))
        new_lp = dist.log_prob(actions)

        ratio = torch.exp(new_lp - old_logp.detach())
        s1 = ratio * ret
        s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * ret
        loss = -torch.min(s1, s2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def evaluate_agent(env, trainer, episodes=100):
    solved, steps = 0, []
    for _ in range(episodes):
        state = env.reset()
        done = False
        cnt = 0
        while not done and cnt < 1000:
            st = torch.tensor(state[None], dtype=torch.long, device=device)
            logits = trainer.model(st)
            dist = Categorical(logits=logits.squeeze(-1))
            act = dist.sample().item()
            f, t = divmod(act, env.num_tubes)
            state, _, done, _ = env.step((f, t))
            cnt += 1
        if env.check_win():
            solved += 1
            steps.append(cnt)
    wr = solved / episodes
    return wr, np.mean(steps) if steps else float('inf')


def main():
    curriculum = [(5, 3, 4), (8, 5, 4), (12, 8, 5)]
    episodes_per_stage = 500

    print("🚀 WATER SORT PPO WITH CURRICULUM")
    for si, (nt, nc, cap) in enumerate(curriculum, 1):
        print(f"\nStage {si}/{len(curriculum)}: {nt} tubes, {nc} colors, cap {cap}")
        env = WaterSortEnv(nt, cap)
        model = WaterSortGNN(nt, cap)
        trainer = PPOTrainer(model)

        for ep in range(1, episodes_per_stage+1):
            state = env.reset()
            done = False
            ep_s, ep_a, ep_r, ep_lp = [], [], [], []
            while not done:
                st = torch.tensor(state[None], dtype=torch.long, device=device)
                logits = model(st)
                dist = Categorical(logits=logits.squeeze(-1))
                a = dist.sample()
                lp = dist.log_prob(a)
                idx = a.item()
                f, t = divmod(idx, nt)

                ns, rew, done, _ = env.step((f, t))
                ep_s.append(st)
                ep_a.append(a)
                ep_r.append(rew)
                ep_lp.append(lp)
                state = ns

            loss = trainer.update(torch.cat(ep_s), torch.stack(ep_a), ep_r, torch.stack(ep_lp))
            if ep % 100 == 0:
                wr, avg_s = evaluate_agent(env, trainer)
                print(f"Stage{si} Ep{ep}/{episodes_per_stage} | WinRate:{wr*100:.1f}% | AvgSteps:{avg_s:.1f}")

    print("\n🎉 TRAINING COMPLETE!")
    print("="*40)

if __name__ == "__main__":
    main()