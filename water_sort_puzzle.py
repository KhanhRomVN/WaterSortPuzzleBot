# To install dependencies:
# !pip install torch numpy torch_geometric GPUtil psutil

import os
import sys
import random
import time
import logging
import numpy as np
from collections import deque
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_sort_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class WaterSortEnv:
    """Optimized Water Sort Puzzle Environment"""
    
    def __init__(self, num_tubes=5, tube_capacity=4, max_moves=50):
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.max_moves = max_moves
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.num_colors = self.num_tubes - 2
        self.tubes = [[] for _ in range(self.num_tubes)]
        self.moves = 0
        self.game_over = False
        self.completed_tubes = 0
        
        # Generate random but solvable initial state
        colors = list(range(self.num_colors))
        color_pool = colors * self.tube_capacity
        random.shuffle(color_pool)
        
        # Distribute colors to tubes
        for i in range(self.num_colors):
            start_idx = i * self.tube_capacity
            self.tubes[i] = color_pool[start_idx:start_idx + self.tube_capacity]
        
        self._update_completed_tubes()
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        state = np.zeros((self.num_tubes, self.tube_capacity + 2), dtype=np.float32)
        
        for tube_idx, tube in enumerate(self.tubes):
            # Tube completion status (1 if completed, 0 otherwise)
            completed = 1.0 if self._is_tube_completed(tube) else 0.0
            state[tube_idx, 0] = completed
            
            # Tube fill level (normalized)
            fill_level = len(tube) / self.tube_capacity
            state[tube_idx, 1] = fill_level
            
            # Color encoding
            for level, color in enumerate(tube):
                state[tube_idx, level + 2] = (color + 1) / (self.num_colors + 1)
                
        return state
    
    def _is_tube_completed(self, tube):
        """Check if tube is completed (all same color and full)"""
        if len(tube) != self.tube_capacity:
            return False
        return all(color == tube[0] for color in tube)
    
    def _update_completed_tubes(self):
        """Update count of completed tubes"""
        self.completed_tubes = sum(1 for tube in self.tubes if self._is_tube_completed(tube))
    
    def is_valid_move(self, from_tube, to_tube):
        """Check if move is valid"""
        # Basic checks
        if from_tube == to_tube:
            return False
        if not self.tubes[from_tube]:
            return False
        if len(self.tubes[to_tube]) >= self.tube_capacity:
            return False
            
        # Color matching check
        from_color = self.tubes[from_tube][-1]
        to_color = self.tubes[to_tube][-1] if self.tubes[to_tube] else None
        
        return to_color is None or from_color == to_color
    
    def get_valid_moves(self):
        """Get all valid moves in current state"""
        valid_moves = []
        for from_tube in range(self.num_tubes):
            for to_tube in range(self.num_tubes):
                if self.is_valid_move(from_tube, to_tube):
                    valid_moves.append((from_tube, to_tube))
        return valid_moves
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        from_tube, to_tube = action
        reward = 0
        info = {}
        
        # Check if move is valid
        if not self.is_valid_move(from_tube, to_tube):
            reward = -0.1
            self.moves += 1
        else:
            # Execute move
            color = self.tubes[from_tube].pop()
            self.tubes[to_tube].append(color)
            self.moves += 1
            
            # Calculate rewards
            reward = self._calculate_reward(from_tube, to_tube)
            
            # Update completed tubes
            old_completed = self.completed_tubes
            self._update_completed_tubes()
            
            # Bonus for new completed tube
            if self.completed_tubes > old_completed:
                reward += 2.0
                info['tube_completed'] = True
        
        # Check terminal conditions
        done = self.check_win() or self.moves >= self.max_moves
        
        if self.check_win():
            reward += 10.0
            info['win'] = True
        elif self.moves >= self.max_moves:
            reward -= 5.0
            info['max_moves_reached'] = True
            
        self.game_over = done
        info['valid_move'] = self.is_valid_move(from_tube, to_tube)
        info['moves'] = self.moves
        info['completed_tubes'] = self.completed_tubes
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, from_tube, to_tube):
        """Calculate reward for move"""
        reward = -0.01  # Small penalty for each move
        
        # Reward for good moves
        to_tube_after = len(self.tubes[to_tube])
        
        # Reward for creating empty tubes
        if len(self.tubes[from_tube]) == 0:
            reward += 0.1
            
        # Reward for filling tubes with same color
        if to_tube_after > 1 and len(self.tubes[to_tube]) > 0:
            # Kiểm tra tất cả viên bi trong to_tube có cùng màu không
            if all(c == self.tubes[to_tube][0] for c in self.tubes[to_tube]):
                reward += 0.05 * to_tube_after
            
        return reward
    
    def check_win(self):
        """Check if puzzle is solved"""
        return all(self._is_tube_completed(tube) or len(tube) == 0 for tube in self.tubes)
    
    def render(self):
        """Render current state (for debugging)"""
        print(f"\nMoves: {self.moves}/{self.max_moves}")
        for i, tube in enumerate(self.tubes):
            status = "✓" if self._is_tube_completed(tube) else " "
            print(f"Tube {i:2d} [{status}]: {tube}")

class AttentionPooling(nn.Module):
    """Attention-based pooling for graph nodes"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: [batch_size, num_nodes, hidden_dim]
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn_weights * x, dim=1)

class WaterSortPolicyNetwork(nn.Module):
    """Enhanced Policy Network with Graph Attention and Transformer"""
    
    def __init__(self, num_tubes, tube_capacity, hidden_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.hidden_dim = hidden_dim
        
        # State processing
        state_dim = tube_capacity + 2  # +2 for completion status and fill level
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer encoder for tube interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_dim)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        batch_size = state.shape[0]
        
        # Encode each tube separately
        tube_embeddings = self.state_encoder(state.view(-1, state.shape[-1]))
        tube_embeddings = tube_embeddings.view(batch_size, self.num_tubes, self.hidden_dim)
        
        # Apply transformer
        transformer_out = self.transformer(tube_embeddings)
        
        # Global context
        global_context = self.attention_pool(transformer_out)
        
        # Policy logits - tạo logits cho TẤT CẢ các cặp tube (bao gồm cả invalid moves)
        policy_inputs = []
        for i in range(self.num_tubes):
            for j in range(self.num_tubes):
                # Concatenate: from_tube, to_tube, global_context
                pair_embedding = torch.cat([
                    transformer_out[:, i],
                    transformer_out[:, j], 
                    global_context
                ], dim=-1)
                policy_inputs.append(pair_embedding)
        
        policy_inputs = torch.stack(policy_inputs, dim=1)  # [batch_size, num_tubes*num_tubes, hidden_dim*3]
        policy_logits = self.policy_head(policy_inputs).squeeze(-1)  # ✅ [batch_size, num_tubes*num_tubes] - giờ mới đúng!
        
        # Value estimate
        state_value = self.value_head(global_context).squeeze(-1)
        
        return policy_logits, state_value

class PPOTrainer:
    """Enhanced PPO Trainer with GAE and advanced logging"""
    
    def __init__(self, model, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5):
        
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Tracking
        self.training_step = 0
        self.best_win_rate = 0
        self.metrics = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            'explained_variance': []
        }
    
    def compute_advantages(self, rewards, values, dones, next_value):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0
        next_value = next_value.detach()
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, device=device)
    
    def update(self, batch):
        """Update model with PPO"""
        states, actions, old_log_probs, returns, advantages = batch
        
        # Forward pass
        policy_logits, values = self.model(states)
        dist = Categorical(logits=policy_logits)
        
        # New log probs and entropy
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Policy loss (PPO clip)
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_pred_clipped = values + (values - values).clamp(-self.clip_epsilon, self.clip_epsilon)
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.training_step += 1
        
        # Metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
            explained_var = 1 - (returns - values).var() / returns.var()
            
        # Update metrics
        self.metrics['loss'].append(loss.item())
        self.metrics['policy_loss'].append(policy_loss.item())
        self.metrics['value_loss'].append(value_loss.item())
        self.metrics['entropy'].append(entropy.item())
        self.metrics['approx_kl'].append(approx_kl)
        self.metrics['clip_fraction'].append(clip_fraction)
        self.metrics['explained_variance'].append(explained_var.item())
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'grad_norm': grad_norm.item(),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'explained_variance': explained_var.item()
        }

class ExperienceCollector:
    """Collect and process experiences for PPO"""
    
    def __init__(self, num_tubes, gamma=0.99, gae_lambda=0.95):
        self.num_tubes = num_tubes
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add_experience(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def process_trajectory(self, last_value=0):
        """Process trajectory and compute returns and advantages"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # Compute returns
        returns = []
        G = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        
        # Compute advantages
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(device)
        actions_t = torch.LongTensor(self.actions).to(device)
        returns_t = torch.FloatTensor(returns).to(device)
        advantages_t = torch.FloatTensor(advantages).to(device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(device)
        
        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        self.reset()
        
        return states_t, actions_t, old_log_probs_t, returns_t, advantages_t

class CurriculumManager:
    """Manage curriculum learning with adaptive difficulty"""
    
    def __init__(self):
        self.levels = [
            {'tubes': 4, 'colors': 2, 'capacity': 4, 'max_moves': 20, 'target_win_rate': 0.8},
            {'tubes': 5, 'colors': 3, 'capacity': 4, 'max_moves': 30, 'target_win_rate': 0.7},
            {'tubes': 6, 'colors': 4, 'capacity': 4, 'max_moves': 40, 'target_win_rate': 0.6},
            {'tubes': 8, 'colors': 6, 'capacity': 4, 'max_moves': 50, 'target_win_rate': 0.5},
            {'tubes': 10, 'colors': 8, 'capacity': 4, 'max_moves': 60, 'target_win_rate': 0.4},
        ]
        self.current_level = 0
        self.win_rates = deque(maxlen=100)
    
    def get_current_config(self):
        return self.levels[self.current_level]
    
    def update_level(self, win_rate):
        """Update current level based on performance"""
        self.win_rates.append(win_rate)
        avg_win_rate = np.mean(self.win_rates)
        current_target = self.levels[self.current_level]['target_win_rate']
        
        if avg_win_rate >= current_target and self.current_level < len(self.levels) - 1:
            self.current_level += 1
            logger.info(f"🎯 Advancing to level {self.current_level + 1}")
            return True
        elif avg_win_rate < current_target * 0.5 and self.current_level > 0:
            self.current_level -= 1
            logger.info(f"🔙 Regressing to level {self.current_level + 1}")
            return True
        
        return False

def setup_logging_and_checkpoints():
    """Setup logging directories and checkpoint system"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"training_logs_{timestamp}"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return log_dir, checkpoint_dir

def evaluate_policy(env, model, num_episodes=10):
    """Evaluate current policy"""
    wins = 0
    total_rewards = []
    total_moves = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        moves = 0
        done = False
        
        while not done and moves < env.max_moves:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                policy_logits, _ = model(state_t)
                dist = Categorical(logits=policy_logits)
                action = dist.sample().item()
            
            # Convert action index to tube pair
            valid_moves = env.get_valid_moves()
            if valid_moves:
                action_idx = action % len(valid_moves)
                from_tube, to_tube = valid_moves[action_idx]
            else:
                # No valid moves, choose random
                from_tube, to_tube = random.choice([(i, j) for i in range(env.num_tubes) 
                                                  for j in range(env.num_tubes) if i != j])
            
            state, reward, done, info = env.step((from_tube, to_tube))
            episode_reward += reward
            moves += 1
        
        if env.check_win():
            wins += 1
        total_rewards.append(episode_reward)
        total_moves.append(moves)
    
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_moves = np.mean(total_moves)
    
    return win_rate, avg_reward, avg_moves

def train_water_sort_agent():
    """Main training function"""
    
    # Setup
    log_dir, checkpoint_dir = setup_logging_and_checkpoints()
    curriculum = CurriculumManager()
    
    # Training parameters
    total_training_steps = 100000
    eval_interval = 1000
    save_interval = 5000
    batch_size = 64
    
    # Initialize model and trainer
    current_config = curriculum.get_current_config()
    model = WaterSortPolicyNetwork(
        num_tubes=current_config['tubes'],
        tube_capacity=current_config['capacity'],
        hidden_dim=128
    )
    trainer = PPOTrainer(model)
    collector = ExperienceCollector(current_config['tubes'])
    
    # Training metrics
    training_metrics = {
        'step': [],
        'win_rate': [],
        'avg_reward': [],
        'avg_moves': [],
        'level': []
    }
    
    logger.info("🚀 Starting Water Sort Puzzle Training")
    logger.info(f"Initial level: {current_config}")
    
    step = 0
    best_win_rate = 0
    
    while step < total_training_steps:
        # Create environment for current level
        current_config = curriculum.get_current_config()
        env = WaterSortEnv(
            num_tubes=current_config['tubes'],
            tube_capacity=current_config['capacity'],
            max_moves=current_config['max_moves']
        )
        
        # Collect experiences
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while not env.game_over and episode_steps < env.max_moves:
            # Prepare state
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action
            # Get action
            with torch.no_grad():
                policy_logits, value = model(state_t)

                # Mask invalid actions bằng cách set logits = -inf
                action_mask = torch.full((1, env.num_tubes * env.num_tubes), float('-inf'), device=device)
                valid_moves = env.get_valid_moves()

                if valid_moves:
                    for from_t, to_t in valid_moves:
                        action_idx = from_t * env.num_tubes + to_t
                        action_mask[0, action_idx] = 0.0

                    masked_logits = policy_logits + action_mask
                    dist = Categorical(logits=masked_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    # Convert action index to tube pair - FIX: xử lý tensor multi-element
                    action_flat = action.flatten()  # Chuyển về tensor 1D
                    if action_flat.numel() > 1:
                        if step < 10:  # Chỉ log 10 bước đầu để debug
                            logger.debug(f"Action tensor has {action_flat.numel()} elements, taking first element")
                        action_idx = action_flat[0].item()  # Lấy phần tử đầu tiên
                    else:
                        action_idx = action_flat.item()  # Lấy trực tiếp nếu là scalar
                    from_tube = action_idx // env.num_tubes
                    to_tube = action_idx % env.num_tubes
                    
                    # ✅ GÁN LẠI action và log_prob thành scalar tensor
                    action = torch.tensor(action_idx, device=device)
                    log_prob_flat = log_prob.flatten()
                    log_prob = log_prob_flat[0] if log_prob_flat.numel() > 1 else log_prob
                else:
                    # Fallback to random move - FIX: tạo action đúng cách
                    from_tube, to_tube = random.choice([(i, j) for i in range(env.num_tubes)
                                                    for j in range(env.num_tubes) if i != j])
                    action_idx = from_tube * env.num_tubes + to_tube
                    action = torch.tensor(action_idx, device=device)  # ✅ Sửa: tạo scalar tensor
                    log_prob = torch.tensor(0.0, device=device)      # ✅ Sửa: tạo scalar tensor
                
                # Take action
                next_state, reward, done, info = env.step((from_tube, to_tube))
                
                # Store experience
                action_value = action.item()
                log_prob_value = log_prob.item()
            
            collector.add_experience(
                state, action_value, reward,
                value.item(), log_prob_value, done
            )
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            step += 1
            
            # Train if we have enough experiences
            if len(collector.states) >= batch_size:
                batch = collector.process_trajectory()
                metrics = trainer.update(batch)
                
                if step % 100 == 0:
                    logger.info(f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                               f"Policy Loss={metrics['policy_loss']:.4f}, "
                               f"Value Loss={metrics['value_loss']:.4f}")
            
            # Evaluation and logging
            if step % eval_interval == 0:
                win_rate, avg_reward, avg_moves = evaluate_policy(env, model)
                
                # Update curriculum
                curriculum.update_level(win_rate)
                
                # Save metrics
                training_metrics['step'].append(step)
                training_metrics['win_rate'].append(win_rate)
                training_metrics['avg_reward'].append(avg_reward)
                training_metrics['avg_moves'].append(avg_moves)
                training_metrics['level'].append(curriculum.current_level)
                
                logger.info(f"📊 Evaluation at step {step}: "
                           f"Win Rate={win_rate:.3f}, "
                           f"Avg Reward={avg_reward:.2f}, "
                           f"Avg Moves={avg_moves:.1f}, "
                           f"Level={curriculum.current_level + 1}")
                
                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    model_path = os.path.join(checkpoint_dir, f"best_model_{step}.pth")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'step': step,
                        'win_rate': win_rate,
                        'metrics': training_metrics
                    }, model_path)
                    logger.info(f"💾 New best model saved: {model_path}")
            
            # Save checkpoint
            if step % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'step': step,
                    'metrics': training_metrics,
                    'trainer_metrics': trainer.metrics
                }, checkpoint_path)
                
                # Save training metrics
                metrics_path = os.path.join(log_dir, "training_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(training_metrics, f, indent=2)
                
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    logger.info("🎉 Training completed!")
    
    # Final evaluation
    final_win_rate, final_reward, final_moves = evaluate_policy(env, model, num_episodes=100)
    logger.info(f"🏆 Final Performance: "
               f"Win Rate={final_win_rate:.3f}, "
               f"Avg Reward={final_reward:.2f}, "
               f"Avg Moves={final_moves:.1f}")
    
    return model, training_metrics

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        model, metrics = train_water_sort_agent()
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise
    
##