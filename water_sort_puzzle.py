# To install dependencies:
# !pip install torch numpy torch_geometric GPUtil psutil

import sys
import random
import time
import os
import gc
import psutil  # Added for CPU monitoring

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import GPUtil

from torch_geometric.nn import GINConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class WaterSortEnv:
    """Optimized Water Sort environment with memory management."""
    def __init__(self, num_tubes=5, tube_capacity=4):
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.reset()

    def reset(self):
        self.COLORS = list(range(self.num_tubes - 2))
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
        # Use int16 for memory savings
        state = np.zeros((self.num_tubes, self.tube_capacity + 1), dtype=np.int16)
        for ti, tube in enumerate(self.tubes):
            completed = 1 if (len(tube) == self.tube_capacity
                              and all(c == tube[0] for c in tube)) else 0
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
        if f < 0 or f >= self.num_tubes or t < 0 or t >= self.num_tubes:
            return self._get_state(), -0.1, self.game_over, {}
        self.moves += 1
        reward = -0.01

        if not self.is_valid_move(f, t):
            return self._get_state(), -0.05, self.game_over, {}

        color = self.tubes[f].pop()
        self.tubes[t].append(color)

        # Reward for completing a tube
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
            # Non-empty tube must be full and monochromatic
            if tube:
                if len(tube) != self.tube_capacity or any(c != tube[0] for c in tube):
                    return False
        return True

class WaterSortGNN(nn.Module):
    """Optimized GNN+Transformer policy network with precomputed edges."""
    def __init__(self, num_tubes, tube_capacity, hidden_dim=128, gnn_layers=3, transformer_layers=2):
        super().__init__()
        vocab = num_tubes + 2
        self.embedding = nn.Embedding(vocab, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gnn_layers.append(GINConv(mlp))

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=transformer_layers
        )
        self.action_head = nn.Linear(hidden_dim * 2, 1)
        self.num_tubes = num_tubes
        self.edge_index = self._build_edge_index(num_tubes)

    def _build_edge_index(self, T):
        idx = []
        for i in range(T):
            for j in range(T):
                if i != j:
                    idx.append([i, j])
        ei = torch.tensor(idx, dtype=torch.long).t().contiguous()
        return ei.to(device)

    def forward(self, state):
        B, T, L = state.shape
        state = state.long().to(device)

        # embedding
        x = self.embedding(state)  # (B, T, L, H)
        tube_repr = x.mean(dim=2)  # (B, T, H)
        flat = tube_repr.view(-1, tube_repr.size(-1))  # (B*T, H)

        edge_index = self.edge_index

        # GNN layers
        for gnn in self.gnn_layers:
            flat = gnn(flat, edge_index)
            flat = torch.relu(flat)

        tube_repr = flat.view(B, T, -1)
        tube_repr = self.transformer(tube_repr)  # (B, T, H)

        # compute all (i→j) pairs
        actions = []
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                pair = torch.cat([tube_repr[:, i], tube_repr[:, j]], dim=1)
                actions.append(self.action_head(pair))
        return torch.cat(actions, dim=1)

class PPOTrainer:
    """Optimized PPO trainer with memory management."""
    def __init__(self, model, lr=3e-4, gamma=0.99, clip_eps=0.2):
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.loss_history = []
        self.kl_divergence = []
        self.gradient_norm = 0.0
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def update(self, states, actions, rewards, old_logp):
        # compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret = torch.tensor(returns, device=device, dtype=torch.float32)
        
        # Handle single-element case
        if len(ret) > 1:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        else:
            ret = (ret - ret.mean())  # Already centered, no division needed

        # Check for NaN in returns
        if torch.isnan(ret).any() or torch.isinf(ret).any():
            print("⚠️ Non-finite values in returns! Skipping update.")
            return 0.0  # Skip this update

        # Forward pass
        logits = self.model(states)
        dist = Categorical(logits=logits.squeeze(-1))
        new_lp = dist.log_prob(actions)

        # KL for monitoring
        with torch.no_grad():
            kl = (old_logp.exp() * (old_logp - new_lp)).mean()
            self.kl_divergence.append(kl.item())

        ratio = torch.exp(new_lp - old_logp.detach())
        surr1 = ratio * ret
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ret
        loss = -torch.min(surr1, surr2).mean()
        self.loss_history.append(loss.item())

        # Optimize with gradient handling
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before clipping
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grads):
            print("⚠️ Non-finite gradients detected! Skipping update.")
            self.optimizer.zero_grad()
            return loss.item()

        # Clip gradients with error handling
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=0.5,
                error_if_nonfinite=True
            ).item()
        except RuntimeError as e:
            if 'non-finite' in str(e):
                print("⚠️ Non-finite gradients during clipping! Skipping update.")
                self.optimizer.zero_grad()
                return loss.item()
            raise

        self.gradient_norm = total_norm
        self.optimizer.step()

        # Free memory
        del logits, dist, new_lp, ratio, surr1, surr2, grads
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        return loss.item()

def evaluate_agent(env, trainer, episodes=10):
    solved, steps, rewards = 0, [], []
    for _ in range(episodes):
        state = env.reset()
        done, cnt, ep_reward = False, 0, 0
        while not done and cnt < env.max_moves:
            st = torch.tensor(state[None], dtype=torch.long, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = trainer.model(st)
                dist = Categorical(logits=logits.squeeze(-1))
                act = dist.sample().item()
            f, t = divmod(act, env.num_tubes)
            state, r, done, _ = env.step((f, t))
            ep_reward += r
            cnt += 1
        rewards.append(ep_reward)
        if env.check_win():
            solved += 1
            steps.append(cnt)
    wr = solved / episodes
    avg_steps = np.mean(steps) if steps else float('inf')
    return wr, avg_steps, np.mean(rewards)

def monitor_resources():
    """Print detailed resource usage information."""
    # CPU monitoring
    cpu_percent = psutil.cpu_percent()
    cpu_mem = psutil.virtual_memory().percent
    
    print(f"CPU Usage: {cpu_percent}% | Memory Usage: {cpu_mem}%")
    
    # GPU monitoring
    if torch.cuda.is_available():
        for i, gpu in enumerate(GPUtil.getGPUs()):
            print(f"GPU {i}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    else:
        print("No GPU available")

def main():
    curriculum = [
        # tubes, colors, capacity, max_moves, hidden_dim, gnn_layers, trans_layers, batch_size
        (5, 3, 4, 50, 128, 3, 2, 64),
        (8, 5, 4, 100, 96, 2, 1, 32),
        (12, 8, 5, 150, 64, 2, 1, 16),
    ]
    episodes_per_stage = 500

    print("=" * 60)
    print("🚀 WATER SORT PPO WITH MEMORY OPTIMIZATION")
    print(f"Device: {device}")
    print("=" * 60)

    log_dir = "water_sort_logs"
    os.makedirs(log_dir, exist_ok=True)

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

    # Add resource monitoring interval
    MONITOR_INTERVAL = 5  # minutes
    last_monitor_time = time.time()
    
    for si, (nt, nc, cap, max_moves, hidden_dim, gnn_layers, trans_layers, batch_size) in enumerate(curriculum, 1):
        stage_start = time.time()
        print(f"\n{'='*50}")
        print(f"🔥 STAGE {si}/{len(curriculum)}: Tubes={nt}, Colors={nc}, Capacity={cap}")
        print(f"Batch Size: {batch_size} | Hidden Dim: {hidden_dim}")
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

        monitor_resources()

        env = WaterSortEnv(nt, cap)
        env.max_moves = max_moves
        model = WaterSortGNN(nt, cap, hidden_dim, gnn_layers, trans_layers)
        trainer = PPOTrainer(model)

        log_file = os.path.join(log_dir, f"stage_{si}_log.csv")
        with open(log_file, 'w') as f:
            f.write("episode,total_reward,episode_length,avg_reward,loss,win_rate,avg_steps,eval_reward,"
                    "kl_div,grad_norm,time,cpu_usage,cpu_mem,gpu_mem\n")

        # Adaptive batch size management with OOM retry limit
        max_oom_retries = 5
        oom_retries = 0

        # Adaptive batch size management
        current_batch_size = batch_size
        oom_occurred = False

        for ep in range(1, episodes_per_stage + 1):
            ep_start = time.time()
            state = env.reset()
            done = False
            total_reward = 0
            states_batch, actions_batch, rewards_batch, logp_batch = [], [], [], []
            step_count = 0

            while not done:
                st = torch.tensor(state[None], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(st)
                    dist = Categorical(logits=logits.squeeze(-1))
                    a = dist.sample()
                    lp = dist.log_prob(a)

                idx = a.item()
                f, t = divmod(idx, nt)
                ns, rew, done, _ = env.step((f, t))

                states_batch.append(st)
                actions_batch.append(a)
                rewards_batch.append(rew)
                logp_batch.append(lp)
                total_reward += rew
                state = ns
                step_count += 1

                # Resource monitoring
                current_time = time.time()
                if current_time - last_monitor_time > MONITOR_INTERVAL * 60:
                    monitor_resources()
                    last_monitor_time = current_time

                # Adaptive batch processing with OOM protection
                if len(states_batch) >= current_batch_size or done:
                    if states_batch:
                        try:
                            s_t = torch.cat(states_batch)
                            a_t = torch.stack(actions_batch)
                            r_t = rewards_batch
                            lp_t = torch.stack(logp_batch)

                            loss = trainer.update(s_t, a_t, r_t, lp_t)

                            states_batch, actions_batch, rewards_batch, logp_batch = [], [], [], []
                            oom_occurred = False
                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                new_size = max(1, current_batch_size // 2)
                                print(f"⚠️ OOM at episode {ep}! Reducing batch size {current_batch_size}→{new_size}")

                                # Enforce retry limit
                                if oom_retries >= max_oom_retries:
                                    raise RuntimeError("OOM retries exceeded")
                                oom_retries += 1

                                # Apply reduction and reset batch
                                current_batch_size = new_size
                                states_batch, actions_batch, rewards_batch, logp_batch = [], [], [], []
                                torch.cuda.empty_cache()
                                gc.collect()
                                oom_occurred = True
                                # Exit this batch to start a fresh episode
                                break
                            else:
                                raise

            episode_len = step_count or 1
            avg_r = total_reward / episode_len

            # Periodic evaluation
            eval_wr, eval_steps, eval_r = (0, 0, 0)
            if ep % 50 == 0 or ep == episodes_per_stage:
                eval_wr, eval_steps, eval_r = evaluate_agent(env, trainer, episodes=5)

            time_taken = time.time() - ep_start
            
            # Resource usage metrics
            cpu_usage = psutil.cpu_percent()
            cpu_mem = psutil.virtual_memory().percent
            gpu_mem = GPUtil.getGPUs()[0].memoryUsed if torch.cuda.is_available() else 0

            if ep % 10 == 0 or oom_occurred:
                with open(log_file, 'a') as f:
                    f.write(f"{ep},{total_reward:.2f},{episode_len},{avg_r:.4f},{loss if 'loss' in locals() else 0:.5f},"
                            f"{eval_wr:.3f},{eval_steps:.1f},{eval_r:.2f},"
                            f"{trainer.kl_divergence[-1] if trainer.kl_divergence else 0:.4f},{trainer.gradient_norm:.2f},{time_taken:.1f},"
                            f"{cpu_usage},{cpu_mem},{gpu_mem}\n")

                print(f"Stage{si} Ep{ep:03d}/{episodes_per_stage} | "
                      f"R:{total_reward:6.2f} | Len:{episode_len:2d} | "
                      f"AvgR:{avg_r:7.4f} | Loss:{loss if 'loss' in locals() else 0:7.4f} | "
                      f"KL:{trainer.kl_divergence[-1] if trainer.kl_divergence else 0:6.4f} | "
                      f"Grad:{trainer.gradient_norm:6.2f} | "
                      f"Time:{time_taken:4.1f}s | CPU:{cpu_usage}% | Mem:{cpu_mem}% | GPU:{gpu_mem}MB")

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stage_time = time.time() - stage_start
        print(f"\n✅ Stage {si} completed in {stage_time/60:.1f} minutes")
        print(f"Log file saved to: {log_file}")

    print("\n🎉 TRAINING COMPLETE!")
    print(f"All logs saved to: {os.path.abspath(log_dir)}")
    print("="*60)

if __name__ == "__main__":
    main()