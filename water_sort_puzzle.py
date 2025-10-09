# To install dependencies:
# !pip install torch numpy tqdm

import os
import sys
import random
import time
import logging
import numpy as np
from collections import deque
import json
from datetime import datetime
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

# Setup logging - tương thích với Jupyter/Colab
warnings.filterwarnings('ignore')

# Disable các log spam từ libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

# Tạo logger đơn giản không conflict với notebook
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Chỉ log vào file, không log ra console
try:
    file_handler = logging.FileHandler('water_sort_training.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
except:
    pass  # Bỏ qua nếu không thể tạo file log

# Helper function cho console output
def print_info(message):
    """Print important info to console"""
    print(f"\n{message}")
    try:
        logger.info(message)
    except:
        pass  # Bỏ qua nếu logger fail

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def _calculate_tube_purity(self, tube):
        """Tính độ tinh khiết của tube (0.0 - 1.0)"""
        if len(tube) == 0:
            return 0.0
        if len(tube) == 1:
            return 1.0
        
        same_color = sum(1 for c in tube if c == tube[0])
        return same_color / len(tube)
    
    def _calculate_reward(self, from_tube, to_tube):
        """Improved reward function khuyến khích optimal path"""
        reward = 0.0
        
        # 1. Strong penalty cho mỗi move (khuyến khích tối ưu số bước)
        reward -= 0.5
        
        # 2. Huge bonus cho completed tube
        old_completed = self.completed_tubes
        self._update_completed_tubes()
        if self.completed_tubes > old_completed:
            reward += 10.0
        
        # 3. Progressive purity bonus (khuyến khích tạo pure tubes sớm)
        if len(self.tubes[to_tube]) >= 2:
            if all(color == self.tubes[to_tube][0] for color in self.tubes[to_tube]):
                tube_length = len(self.tubes[to_tube])
                # Bonus tăng exponentially: 1, 2, 4, 8
                reward += 2 ** (tube_length - 1) * 0.5
        
        # 4. Heavy penalty cho phá vỡ pure tube
        if len(self.tubes[from_tube]) >= 2:
            if all(color == self.tubes[from_tube][0] for color in self.tubes[from_tube]):
                reward -= 3.0
        
        # 5. Bonus cho việc sử dụng empty tubes hiệu quả
        if len(self.tubes[to_tube]) == 0:
            # Chỉ bonus nếu move từ mixed tube
            from_tube_purity = self._calculate_tube_purity(self.tubes[from_tube])
            if from_tube_purity < 1.0:
                reward += 0.5
        
        # 6. Win bonus tỉ lệ nghịch với số moves
        if self.check_win():
            efficiency_bonus = (self.max_moves - self.moves) / self.max_moves * 20.0
            reward += efficiency_bonus
        
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
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn_weights * x, dim=1)

class WaterSortPolicyNetwork(nn.Module):
    """Enhanced Policy Network with Graph Attention and Transformer"""
    
    def __init__(self, num_tubes, tube_capacity, hidden_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.hidden_dim = hidden_dim
        
        # State processing đơn giản hơn, tăng eps cho LayerNorm
        state_dim = tube_capacity + 2
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),  # Tăng eps
            nn.ReLU(),
            nn.Dropout(0.05),  # Giảm dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),  # Tăng eps
            nn.ReLU(),
            nn.Dropout(0.05)  # Giảm dropout
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
        
        # Robust normalization với eps lớn hơn
        state = torch.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Sử dụng running statistics thay vì batch norm
        eps = 1e-3  # Tăng eps lên đáng kể từ 1e-6
        state_flat = state.view(-1, state.shape[-1])
        mean = state_flat.mean()
        std = state_flat.std() + eps
        state = (state - mean) / std
        state = torch.clamp(state, -5.0, 5.0)  # Giảm clamp range từ 10.0
        
        # Encode each tube separately
        tube_embeddings = self.state_encoder(state.view(-1, state.shape[-1]))
        tube_embeddings = tube_embeddings.view(batch_size, self.num_tubes, self.hidden_dim)
        
        # Check for NaN sau encoder (silent fix)
        if torch.isnan(tube_embeddings).any():
            tube_embeddings = torch.where(torch.isnan(tube_embeddings), 
                                         torch.zeros_like(tube_embeddings), 
                                         tube_embeddings)
        
        # Apply transformer
        transformer_out = self.transformer(tube_embeddings)
        
        # Check for NaN sau transformer (silent fix)
        if torch.isnan(transformer_out).any():
            transformer_out = torch.where(torch.isnan(transformer_out),
                                         torch.zeros_like(transformer_out),
                                         transformer_out)
        
        # Global context
        global_context = self.attention_pool(transformer_out)
        
        # Policy logits với stability
        policy_inputs = []
        for i in range(self.num_tubes):
            for j in range(self.num_tubes):
                pair_embedding = torch.cat([
                    transformer_out[:, i],
                    transformer_out[:, j], 
                    global_context
                ], dim=-1)
                policy_inputs.append(pair_embedding)
        
        policy_inputs = torch.stack(policy_inputs, dim=1)
        policy_logits = self.policy_head(policy_inputs).squeeze(-1)
        
        # Clamp logits để tránh overflow
        policy_logits = torch.clamp(policy_logits, -20.0, 20.0)
        
        # Replace NaN với giá trị nhỏ (silent fix)
        if torch.isnan(policy_logits).any():
            policy_logits = torch.where(torch.isnan(policy_logits),
                                       torch.tensor(-1e9, device=policy_logits.device),
                                       policy_logits)
        
        # Value estimate
        state_value = self.value_head(global_context).squeeze(-1)
        state_value = torch.clamp(state_value, -100.0, 100.0)
        
        return policy_logits, state_value

class PPOTrainer:
    """Enhanced PPO Trainer with GAE"""
    
    def __init__(self, model, lr=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.02,
                 max_grad_norm=1.0):
        
        self.model = model.to(device)
        # Sử dụng AdamW với weight decay riêng biệt và tăng eps
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-4, 
                                      weight_decay=1e-4, betas=(0.9, 0.999))
        
        # Giảm warmup steps để học nhanh hơn
        self.warmup_steps = 1000
        self.current_step = 0
        
        # Thêm adaptive entropy coefficient
        self.initial_entropy_coef = entropy_coef
        self.min_entropy_coef = entropy_coef * 0.3  # Decay xuống 30%
        self.entropy_decay_steps = 50000
        
        # Thêm learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10000,  # Restart mỗi 10k steps
            T_mult=2,   # Double chu kỳ sau mỗi restart
            eta_min=1e-6  # Minimum LR
        )
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.training_step = 0
        self.best_win_rate = 0
        self.nan_count = 0  # Track NaN occurrences
        self.metrics = {
            'loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            'explained_variance': []
        }
    
    def update(self, batch):
        """Update model with PPO"""
        states, actions, old_log_probs, returns, advantages = batch
        
        # Forward pass
        policy_logits, values = self.model(states)
        dist = Categorical(logits=policy_logits)
        
        # New log probs and entropy với stability
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Clamp log probs để tránh overflow khi exp
        new_log_probs = torch.clamp(new_log_probs, -20.0, 20.0)
        old_log_probs = torch.clamp(old_log_probs, -20.0, 20.0)
        
        # Policy loss (PPO clip)
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # Prevent overflow
        ratio = log_ratio.exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_pred_clipped = values + (values - values).clamp(-self.clip_epsilon, self.clip_epsilon)
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        
        # Adaptive entropy coefficient với linear decay
        progress = min(1.0, self.training_step / self.entropy_decay_steps)
        current_entropy_coef = self.initial_entropy_coef - (self.initial_entropy_coef - self.min_entropy_coef) * progress
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - current_entropy_coef * entropy
        
        # Optimization step
        self.optimizer.zero_grad()
        
        # Kiểm tra loss trước khi backward (silent skip)
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'grad_norm': 0.0,
                'approx_kl': 0.0,
                'clip_fraction': 0.0,
                'explained_variance': 0.0,
                'skipped': True
            }
        
        loss.backward()
        
        # Double gradient clipping với monitoring và adaptive LR
        for p in self.model.parameters():
            if p.grad is not None:
                # Detect và fix NaN/Inf gradients
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    p.grad.data.zero_()  # Zero out thay vì skip
                    self.nan_count += 1
        
        # Adaptive gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        total_norm = grad_norm.item()
        
        # Giảm LR tạm thời nếu gradient explode
        if grad_norm > self.max_grad_norm * 2:
            for g in self.optimizer.param_groups:
                g['lr'] *= 0.5
                
        # Skip update nếu gradient quá lớn (tăng threshold)
        if grad_norm > self.max_grad_norm * 10:
            self.nan_count += 1
            return {
                'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                'entropy': 0.0, 'grad_norm': grad_norm.item(),
                'approx_kl': 0.0, 'clip_fraction': 0.0,
                'explained_variance': 0.0, 'skipped': True
            }
        
        # Update learning rate với warmup
        self.update_learning_rate()
        self.optimizer.step()
        self.training_step += 1
        
        # Metrics
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
            explained_var = 1 - (returns - values).var() / returns.var()
            
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

    def update_learning_rate(self):
        """Update learning rate với warmup và scheduler"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = min(1.0, self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults['lr'] * lr_scale
        else:
            # Sử dụng scheduler sau warmup
            self.scheduler.step()
        
        self.current_step += 1

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
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=bool)
        
        # Compute returns với numpy vectorization
        returns = np.zeros_like(rewards)
        G = last_value
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G * (1.0 - float(dones[t]))
            returns[t] = G
        
        # Compute advantages với vectorization
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages[t] = gae
        
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
            {'tubes': 4, 'colors': 2, 'capacity': 4, 'max_moves': 30, 'target_win_rate': 0.55},   # Giảm từ 0.65
            {'tubes': 5, 'colors': 3, 'capacity': 4, 'max_moves': 40, 'target_win_rate': 0.5},    # Giảm từ 0.6
            {'tubes': 6, 'colors': 4, 'capacity': 4, 'max_moves': 50, 'target_win_rate': 0.55},
            {'tubes': 7, 'colors': 5, 'capacity': 4, 'max_moves': 60, 'target_win_rate': 0.5},    # More forgiving
            {'tubes': 8, 'colors': 6, 'capacity': 4, 'max_moves': 70, 'target_win_rate': 0.45},
            {'tubes': 10, 'colors': 8, 'capacity': 4, 'max_moves': 80, 'target_win_rate': 0.4},
        ]
        
        self.current_level = 0
        self.win_rates = deque(maxlen=30)  # Giảm từ 50 để level up nhanh hơn
    
    def get_current_config(self):
        return self.levels[self.current_level]
    
    def update_level(self, win_rate):
        """Update current level based on performance - More stable"""
        self.win_rates.append(win_rate)
        
        # Tăng requirement lên 30 episodes để stable hơn
        if len(self.win_rates) < 30:
            return False
            
        avg_win_rate = np.mean(self.win_rates)
        current_target = self.levels[self.current_level]['target_win_rate']
        
        # Kiểm tra last 20 episodes (tăng từ 10 để tránh noise)
        recent_win_rate = np.mean(list(self.win_rates)[-20:])
        
        # Chuyển level lên CHỈ KHI cả avg và recent đều vượt target
        if (avg_win_rate >= current_target and 
            recent_win_rate >= current_target * 1.05 and 
            self.current_level < len(self.levels) - 1):
            self.current_level += 1
            self.win_rates.clear()
            return True
        # Giảm level khi performance quá thấp
        elif avg_win_rate < current_target * 0.4 and self.current_level > 0:
            self.current_level -= 1
            self.win_rates.clear()
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

def download_model(model_path, training_metrics):
    """Tự động tải xuống model và metrics"""
    try:
        import shutil
        
        # Tạo thư mục download
        download_dir = "downloaded_models"
        os.makedirs(download_dir, exist_ok=True)
        
        # Copy model file
        model_filename = os.path.basename(model_path)
        download_path = os.path.join(download_dir, model_filename)
        shutil.copy2(model_path, download_path)
        
        # Save metrics
        metrics_path = os.path.join(download_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # Create summary file
        summary_path = os.path.join(download_dir, "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("WATER SORT PUZZLE - TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model saved at: {download_path}\n")
            f.write(f"Metrics saved at: {metrics_path}\n\n")
            
            if training_metrics['win_rate']:
                f.write(f"Best Win Rate: {max(training_metrics['win_rate']):.3f}\n")
                f.write(f"Final Win Rate: {training_metrics['win_rate'][-1]:.3f}\n")
                f.write(f"Total Training Steps: {training_metrics['step'][-1]}\n")
        
        print_info(f"✅ Model và metrics đã được tải xuống vào: {download_dir}/")
        print_info(f"   - Model: {model_filename}")
        print_info(f"   - Metrics: training_metrics.json")
        print_info(f"   - Summary: training_summary.txt")
        
        return download_path
        
    except Exception as e:
        print(f"⚠️  Lỗi khi tải xuống model: {e}")
        return None

def load_pretrained_model(model, checkpoint_path):
    """Load pretrained model để fine-tuning"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Trả về thông tin bổ sung nếu có
        info = {
            'step': checkpoint.get('step', 0),
            'win_rate': checkpoint.get('win_rate', 0.0),
            'optimizer_state': checkpoint.get('optimizer_state_dict', None)
        }
        
        print_info(f"✅ Đã load model từ: {checkpoint_path}")
        print_info(f"   - Training step: {info['step']}")
        print_info(f"   - Win rate: {info['win_rate']:.1%}")
        
        return model, info
        
    except FileNotFoundError:
        print_info(f"❌ Không tìm thấy file: {checkpoint_path}")
        return None, None
    except Exception as e:
        print_info(f"❌ Lỗi khi load model: {e}")
        return None, None

def get_version_from_user():
    """Lấy version number từ người dùng"""
    while True:
        try:
            print("\n" + "="*60)
            print("📝 NHẬP VERSION CHO MODEL")
            print("="*60)
            version_input = input("Nhập version number (vd: 1, 2, 3...): ").strip()
            
            # Validate input
            version = int(version_input)
            if version <= 0:
                print("⚠️  Version phải là số nguyên dương. Vui lòng thử lại.")
                continue
            
            # Confirm
            confirm = input(f"Xác nhận version v{version}? (y/n): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                return version
            else:
                print("Hủy. Vui lòng nhập lại.")
                
        except ValueError:
            print("⚠️  Input không hợp lệ. Vui lòng nhập số nguyên.")
        except KeyboardInterrupt:
            print("\n⚠️  Đã hủy nhập version. Sử dụng version mặc định: 1")
            return 1

def evaluate_policy(env, model, num_episodes=10):
    """Evaluate current policy với optimal path metrics"""
    wins = 0
    total_rewards = []
    total_moves = []
    win_moves = []  # Track moves của winning episodes
    
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
                from_tube, to_tube = random.choice([(i, j) for i in range(env.num_tubes) 
                                                  for j in range(env.num_tubes) if i != j])
            
            state, reward, done, info = env.step((from_tube, to_tube))
            episode_reward += reward
            moves += 1
        
        if env.check_win():
            wins += 1
            win_moves.append(moves)
        total_rewards.append(episode_reward)
        total_moves.append(moves)
    
    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_moves = np.mean(total_moves)
    
    # Thêm metrics cho winning episodes
    avg_win_moves = np.mean(win_moves) if win_moves else 0
    min_win_moves = min(win_moves) if win_moves else 0
    
    return win_rate, avg_reward, avg_moves, avg_win_moves, min_win_moves

def transfer_compatible_weights(old_model, new_model):
    """Transfer weights từ old model sang new model (compatible layers only)"""
    try:
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        
        transferred = 0
        for name, param in old_state.items():
            if name in new_state:
                # Chỉ transfer nếu shape compatible
                if param.shape == new_state[name].shape:
                    new_state[name] = param
                    transferred += 1
        
        new_model.load_state_dict(new_state)
        print_info(f"   ✅ Transferred {transferred}/{len(old_state)} layers")
        
    except Exception as e:
        print_info(f"   ⚠️  Transfer weights failed: {e}")
        print_info(f"   Starting fresh training for new level")

def train_water_sort_agent(pretrained_path=None):
    """Main training function với progress bar và fine-tuning support
    
    Args:
        pretrained_path: Đường dẫn đến model pretrained (optional)
    """
    
    # Setup
    log_dir, checkpoint_dir = setup_logging_and_checkpoints()
    curriculum = CurriculumManager()
    
    # Training parameters
    total_training_steps = 100000
    eval_interval = 1000
    
    # Hyperparameters cho PPO - Balanced và stable
    ppo_config = {
        'lr': 5e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.03,     # Tăng từ 0.01 để explore tốt hơn
        'max_grad_norm': 2.0
    }
    save_interval = 5000
    batch_size = 256  # Tăng từ 128 để stable hơn và học hiệu quả
    
    # Thêm gradient accumulation để tăng effective batch size
    gradient_accumulation_steps = 2
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    # Initialize model and trainer - Increased capacity
    current_config = curriculum.get_current_config()
    model = None  # Sẽ được khởi tạo lại khi cần
    old_model = None  # Lưu model cũ để transfer learning
    current_num_tubes = None  # Track số tubes hiện tại
    trainer = None  # Trainer cũng sẽ được tạo khi có model
    
    # Load pretrained model nếu có
    start_step = 0
    if pretrained_path:
        print_info(f"\n🔄 FINE-TUNING MODE")
        print_info(f"Đang load model từ: {pretrained_path}")
        
        # Tạo model tạm để load pretrained weights
        temp_model = WaterSortPolicyNetwork(
            num_tubes=current_config['tubes'],
            tube_capacity=current_config['capacity'],
            hidden_dim=128,
            num_heads=4,
            num_layers=2
        )
        
        model, pretrained_info = load_pretrained_model(temp_model, pretrained_path)
        
        if model is None:
            print_info("⚠️  Không thể load pretrained model. Bắt đầu training từ đầu.")
        else:
            start_step = pretrained_info.get('step', 0)
            current_num_tubes = current_config['tubes']
            print_info(f"✅ Tiếp tục training từ step {start_step}")
            
            # Tạo trainer cho pretrained model
            trainer = PPOTrainer(
                model,
                lr=ppo_config['lr'],
                gamma=ppo_config['gamma'],
                gae_lambda=ppo_config['gae_lambda'],
                clip_epsilon=ppo_config['clip_epsilon'],
                value_coef=ppo_config['value_coef'],
                entropy_coef=ppo_config['entropy_coef'],
                max_grad_norm=ppo_config['max_grad_norm']
            )
            
            # Load optimizer state nếu có
            if pretrained_info and pretrained_info.get('optimizer_state'):
                try:
                    trainer.optimizer.load_state_dict(pretrained_info['optimizer_state'])
                    print_info("✅ Đã restore optimizer state")
                except:
                    print_info("⚠️  Không thể restore optimizer state. Sử dụng optimizer mới.")
    
    # Training metrics
    training_metrics = {
        'step': [],
        'win_rate': [],
        'avg_reward': [],
        'avg_moves': [],
        'level': []
    }
    
    print_info("🚀 BẮT ĐẦU TRAINING WATER SORT PUZZLE")
    print_info(f"📋 Cấu hình ban đầu: {current_config}")
    print_info(f"💻 Device: {device}")
    
    step = 0
    best_win_rate = 0
    best_model_path = None
    
    # Progress bar chính - minimal format
    pbar = tqdm(total=total_training_steps, desc="Training",
                unit="step", ncols=80, colour='green',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                dynamic_ncols=True, leave=True, position=0)
    
    while step < total_training_steps:
        # Create environment for current level
        current_config = curriculum.get_current_config()
        
        # 🔥 Re-create model nếu số tubes thay đổi
        if model is None or current_num_tubes != current_config['tubes']:
            print_info(f"\n🔄 Re-creating model for {current_config['tubes']} tubes...")
            
            old_model = model  # Lưu model cũ
            current_num_tubes = current_config['tubes']
            
            model = WaterSortPolicyNetwork(
                num_tubes=current_config['tubes'],
                tube_capacity=current_config['capacity'],
                hidden_dim=128,
                num_heads=4,
                num_layers=2
            )
            
            # Transfer learning từ model cũ (nếu có)
            if old_model is not None:
                transfer_compatible_weights(old_model, model)
            
            # Re-create trainer với model mới
            trainer = PPOTrainer(
                model,
                lr=ppo_config['lr'],
                gamma=ppo_config['gamma'],
                gae_lambda=ppo_config['gae_lambda'],
                clip_epsilon=ppo_config['clip_epsilon'],
                value_coef=ppo_config['value_coef'],
                entropy_coef=ppo_config['entropy_coef'],
                max_grad_norm=ppo_config['max_grad_norm']
            )
            
            collector = ExperienceCollector(current_config['tubes'])
        
        env = WaterSortEnv(
            num_tubes=current_config['tubes'],
            tube_capacity=current_config['capacity'],
            max_moves=current_config['max_moves']
        )
        
        # Collect experiences
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < env.max_moves and step < total_training_steps:
            # Prepare state
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action
            with torch.no_grad():
                if torch.isnan(state_t).any() or torch.isinf(state_t).any():
                    # Silent reset
                    break
                
                policy_logits, value = model(state_t)
                batch_size = policy_logits.shape[0]
                
                # Mask invalid actions - CHỈ xét valid moves
                action_mask = torch.full_like(policy_logits, float('-inf'))
                valid_moves = env.get_valid_moves()

                if valid_moves:
                    # CHỈ cho phép các action hợp lệ
                    valid_indices = []
                    for from_t, to_t in valid_moves:
                        action_idx = from_t * env.num_tubes + to_t
                        action_mask[0, action_idx] = 0.0
                        valid_indices.append(action_idx)

                    masked_logits = policy_logits + action_mask
                    
                    # Đảm bảo chỉ sample từ valid moves
                    dist = Categorical(logits=masked_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    # Convert action index to tube pair
                    action_idx = action.item()
                    from_tube = action_idx // env.num_tubes
                    to_tube = action_idx % env.num_tubes
                    
                    # Verify move is valid
                    if not env.is_valid_move(from_tube, to_tube):
                        # Fallback to first valid move
                        from_tube, to_tube = valid_moves[0]
                        action_idx = from_tube * env.num_tubes + to_tube
                        action = torch.tensor(action_idx, device=device)
                        log_prob = torch.tensor(0.0, device=device)
                    
                else:
                    # No valid moves - skip với penalty nhỏ
                    from_tube, to_tube = 0, 1  # Default move
                    if env.num_tubes > 1:
                        from_tube, to_tube = 0, 1
                    else:
                        from_tube, to_tube = 0, 0
                    action_idx = from_tube * env.num_tubes + to_tube
                    action = torch.tensor(action_idx, device=device)
                    log_prob = torch.tensor(-1.0, device=device)  # Negative log prob cho invalid moves
            
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
            
            # Update progress bar - chỉ update mỗi 1% (1000 steps), không dùng postfix
            if step % 1000 == 0:
                pbar.update(1000)
            
            # Train with gradient accumulation
            if len(collector.states) >= effective_batch_size:
                batch = collector.process_trajectory()
                
                # Split batch for accumulation
                for i in range(gradient_accumulation_steps):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    
                    mini_batch = tuple(b[start_idx:end_idx] for b in batch)
                    metrics = trainer.update(mini_batch)
            
            # Evaluation and logging - chỉ log khi cần thiết
            if step % eval_interval == 0 and step > 0:
                win_rate, avg_reward, avg_moves, avg_win_moves, min_win_moves = evaluate_policy(env, model)
                
                # Update curriculum
                level_changed = curriculum.update_level(win_rate)
                
                # Save metrics
                training_metrics['step'].append(step)
                training_metrics['win_rate'].append(win_rate)
                training_metrics['avg_reward'].append(avg_reward)
                training_metrics['avg_moves'].append(avg_moves)
                training_metrics['level'].append(curriculum.current_level)
                
                # Tính progress percentage
                progress_pct = (step / total_training_steps) * 100
                
                # Log 1 dòng duy nhất, format rõ ràng
                eval_msg = f"[{progress_pct:5.1f}%] Step {step:,} | "
                eval_msg += f"WR: {win_rate:5.1%} | "
                eval_msg += f"Reward: {avg_reward:6.2f} | "
                eval_msg += f"Moves: {avg_moves:4.1f} | "
                eval_msg += f"WinMoves: {avg_win_moves:4.1f} | "
                eval_msg += f"Best: {min_win_moves:2.0f} | "
                eval_msg += f"Lv{curriculum.current_level+1}"
                
                if level_changed:
                    eval_msg += " | 🎯LEVEL UP!"
                if trainer.nan_count > 0:
                    eval_msg += f" | ⚠️NaN:{trainer.nan_count}"
                    trainer.nan_count = 0
                
                # Write với newline để tách biệt
                print(f"\n{eval_msg}")
                
                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_model_path = os.path.join(checkpoint_dir, f"best_model_{step}.pth")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'step': step,
                        'win_rate': win_rate,
                        'metrics': training_metrics
                    }, best_model_path)
                    print(f"          💾 Saved best model (WR: {win_rate:.1%})")
            
            # Save checkpoint (silent - không log)
            if step % save_interval == 0 and step > 0:
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
    
    # Đóng progress bar và clear console spam
    pbar.close()
    print("\n" * 2)  # Tạo khoảng trống
    
    print_info("="*60)
    print_info("🎉 TRAINING HOÀN TẤT!")
    print_info("="*60)
    
    # Final evaluation
    print_info("\n🔍 Đang đánh giá hiệu suất cuối cùng...")
    final_win_rate, final_reward, final_moves, final_win_moves, final_best_moves = evaluate_policy(env, model, num_episodes=100)
    
    print_info("\n📈 KẾT QUẢ CUỐI CÙNG:")
    print_info(f"   Win Rate: {final_win_rate:.1%}")
    print_info(f"   Avg Reward: {final_reward:.2f}")
    print_info(f"   Avg Moves: {final_moves:.1f}")
    
    # Lấy version từ người dùng
    version = get_version_from_user()
    
    # Save final model với version name
    final_model_filename = f"water-sort-puzzle-v{version}.pth"
    final_model_path = os.path.join(checkpoint_dir, final_model_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'step': step,
        'win_rate': final_win_rate,
        'avg_reward': final_reward,
        'avg_moves': final_moves,
        'metrics': training_metrics,
        'version': version,
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    
    print_info(f"\n💾 Final model đã lưu: {final_model_filename}")
    
    # Tự động tải xuống model
    print_info("\n📥 Đang tải xuống model...")
    download_model(final_model_path, training_metrics)
    
    return model, training_metrics

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print_info("=" * 60)
    print_info("🚀 WATER SORT PUZZLE - PPO TRAINING")
    print_info("=" * 60)
    print_info(f"💻 Device: {device}")
    
    # Hỏi người dùng có muốn fine-tune không
    print("\n" + "="*60)
    print("🔧 CHỌN CHẾ ĐỘ TRAINING")
    print("="*60)
    print("1. Train mới từ đầu")
    print("2. Fine-tune từ model có sẵn")
    
    try:
        mode_choice = input("\nChọn chế độ (1 hoặc 2): ").strip()
        
        pretrained_path = None
        if mode_choice == "2":
            pretrained_path = input("Nhập đường dẫn đến file .pth: ").strip()
            if not os.path.exists(pretrained_path):
                print_info(f"⚠️  File không tồn tại: {pretrained_path}")
                print_info("Chuyển sang chế độ train mới từ đầu.")
                pretrained_path = None
        
        model, metrics = train_water_sort_agent(pretrained_path=pretrained_path)
        print_info("\n✅ HOÀN THÀNH THÀNH CÔNG!")
    except KeyboardInterrupt:
        print_info("\n⚠️  Training bị dừng bởi người dùng")
    except Exception as e:
        print_info(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        raise