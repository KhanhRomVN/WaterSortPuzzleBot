# !pip install torch torchvision torchaudio numpy tqdm
# =============================================================================
# WATER SORT PUZZLE AI - ALPHAZERO STYLE
# Single File Implementation for Google Colab
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
import random
import math
from typing import List, Tuple, Optional
import time
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# =============================================================================
# 1. WATER SORT ENVIRONMENT
# =============================================================================

class WaterSortEnv:
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.num_colors = num_colors
        self.bottle_height = bottle_height
        self.num_bottles = num_bottles
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        # Tạo màu sắc (0 = empty, 1..num_colors = colors)
        colors = list(range(1, self.num_colors + 1)) * (self.bottle_height - 1)
        random.shuffle(colors)
        
        # Phân phối màu vào các chai
        self.bottles = np.zeros((self.num_bottles, self.bottle_height), dtype=int)
        color_idx = 0
        
        for i in range(self.num_colors):
            for j in range(self.bottle_height - 1):
                if color_idx < len(colors):
                    # Tìm chai còn chỗ
                    for bottle_idx in range(self.num_bottles):
                        if np.sum(self.bottles[bottle_idx] > 0) < self.bottle_height - 1:
                            empty_pos = np.where(self.bottles[bottle_idx] == 0)[0][-1]
                            self.bottles[bottle_idx, empty_pos] = colors[color_idx]
                            color_idx += 1
                            break
        
        # Đảm bảo có ít nhất 2 chai trống
        self.bottles[-2:] = 0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        return self.bottles.copy()
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves (from_bottle, to_bottle)"""
        valid_moves = []
        
        for from_idx in range(self.num_bottles):
            for to_idx in range(self.num_bottles):
                if from_idx != to_idx and self._is_valid_move(from_idx, to_idx):
                    valid_moves.append((from_idx, to_idx))
        
        return valid_moves
    
    def _is_valid_move(self, from_idx: int, to_idx: int) -> bool:
        """Check if move is valid"""
        from_bottle = self.bottles[from_idx]
        to_bottle = self.bottles[to_idx]
        
        # Check if source has liquid
        if np.sum(from_bottle > 0) == 0:
            return False
        
        # Check if destination has space
        if np.sum(to_bottle > 0) == self.bottle_height:
            return False
        
        # Get top color from source
        source_top_idx = np.where(from_bottle > 0)[0]
        if len(source_top_idx) == 0:
            return False
        source_top_color = from_bottle[source_top_idx[0]]
        
        # Check if destination is empty or same color
        dest_top_idx = np.where(to_bottle > 0)[0]
        if len(dest_top_idx) == 0:  # Empty bottle
            return True
        dest_top_color = to_bottle[dest_top_idx[0]]
        
        return source_top_color == dest_top_color
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute move and return new state, reward, done, info"""
        from_idx, to_idx = action
        
        if not self._is_valid_move(from_idx, to_idx):
            return self.get_state(), -1, False, {"error": "Invalid move"}
        
        # Perform the pour
        self._pour_liquid(from_idx, to_idx)
        
        # Check if solved
        done = self.is_solved()
        reward = 10.0 if done else -0.1  # Small penalty per move
        
        return self.get_state(), reward, done, {}
    
    def _pour_liquid(self, from_idx: int, to_idx: int):
        """Pour liquid from one bottle to another"""
        from_bottle = self.bottles[from_idx]
        to_bottle = self.bottles[to_idx]
        
        # Get source top color and amount
        source_non_empty = np.where(from_bottle > 0)[0]
        if len(source_non_empty) == 0:
            return
        
        source_top_idx = source_non_empty[0]
        source_color = from_bottle[source_top_idx]
        
        # Count consecutive same color from top
        pour_amount = 1
        for i in range(source_top_idx + 1, len(from_bottle)):
            if from_bottle[i] == source_color:
                pour_amount += 1
            else:
                break
        
        # Calculate available space in destination
        dest_empty = np.where(to_bottle == 0)[0]
        if len(dest_empty) == 0:
            return
        
        available_space = len(dest_empty)
        actual_pour = min(pour_amount, available_space)
        
        # Perform the pour
        for i in range(actual_pour):
            from_pos = source_top_idx + i
            to_pos = dest_empty[-(i+1)]  # Fill from bottom of empty space
            
            self.bottles[to_idx, to_pos] = source_color
            self.bottles[from_idx, from_pos] = 0
    
    def is_solved(self) -> bool:
        """Check if puzzle is solved"""
        for bottle in self.bottles:
            unique_colors = np.unique(bottle[bottle > 0])
            if len(unique_colors) > 1:
                return False
            # Check if bottle is full of same color or empty
            if len(unique_colors) == 1 and np.sum(bottle > 0) != self.bottle_height and np.sum(bottle > 0) != 0:
                return False
        return True
    
    def render(self):
        """Visualize current state"""
        for i, bottle in enumerate(self.bottles):
            print(f"Bottle {i}: {bottle}")

# =============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class WaterSortNet(nn.Module):
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6, 
                 num_res_blocks=5, channels=128):
        super().__init__()
        
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
        
        # Input: (num_colors + 1) x bottle_height x num_bottles
        # +1 for empty representation
        self.conv_input = nn.Conv2d(num_colors + 1, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.conv_policy = nn.Conv2d(channels, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * bottle_height * num_bottles, num_bottles * num_bottles)
        
        # Value head
        self.conv_value = nn.Conv2d(channels, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * bottle_height * num_bottles, 256)
        self.fc_value2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Input shape: (batch, num_colors+1, bottle_height, num_bottles)
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        
        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return policy, value
    
    def preprocess_state(self, state_batch):
        """Convert state batch to neural network input format"""
        batch_size = len(state_batch)
        # One-hot encode: (batch, num_colors+1, height, num_bottles)
        processed = np.zeros((batch_size, self.num_colors + 1, 
                            self.bottle_height, self.num_bottles))
        
        for i, state in enumerate(state_batch):
            for bottle_idx in range(self.num_bottles):
                for height_idx in range(self.bottle_height):
                    color = state[bottle_idx, height_idx]
                    if color == 0:  # Empty
                        processed[i, 0, height_idx, bottle_idx] = 1
                    else:
                        processed[i, color, height_idx, bottle_idx] = 1
        
        return torch.FloatTensor(processed)

# =============================================================================
# 3. MONTE CARLO TREE SEARCH
# =============================================================================

class Node:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior  # Prior probability from neural network
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expanded(self):
        return self.is_expanded

class MCTS:
    def __init__(self, env, model, num_simulations=100, c_puct=1.0):
        self.env = env
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, state, temperature=1.0):
        """Perform MCTS search from given state"""
        search_start = time.time()
        root = Node(state)
        
        # Expand root node
        nn_start = time.time()
        policy, value = self.model(self.model.preprocess_state([state]).to(next(self.model.parameters()).device))
        policy = torch.softmax(policy, dim=1).cpu().detach().numpy()[0]
        value = value.item()
        nn_time = time.time() - nn_start
        
        valid_moves = self.env.get_valid_moves()
        valid_actions = [self._action_to_idx(move) for move in valid_moves]
        
        tqdm.write(f"    MCTS: Found {len(valid_moves)} valid moves, NN inference took {nn_time:.3f}s")
        
        # Mask invalid moves
        policy_mask = np.zeros(len(policy))
        for action_idx in valid_actions:
            policy_mask[action_idx] = 1.0
        
        masked_policy = policy * policy_mask
        if np.sum(masked_policy) > 0:
            masked_policy /= np.sum(masked_policy)
        else:
            # Uniform distribution over valid moves
            masked_policy = policy_mask / np.sum(policy_mask)
        
        root.is_expanded = True
        for action_idx, prob in enumerate(masked_policy):
            if prob > 0:
                move = self._idx_to_action(action_idx)
                root.children[move] = Node(None, root, move, prob)
        
        # Perform simulations
        sim_start = time.time()
        for sim_idx in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Select
            while node.expanded():
                node = self._select_child(node)
                search_path.append(node)
            
            # Expand and evaluate
            if not node.expanded():
                value = self._expand_and_evaluate(node)
            
            # Backpropagate
            self._backpropagate(search_path, value)
            
            # Log every 25 simulations
            if (sim_idx + 1) % 25 == 0:
                elapsed = time.time() - sim_start
                tqdm.write(f"    MCTS: {sim_idx + 1}/{self.num_simulations} simulations ({elapsed:.2f}s)")
        
        sim_time = time.time() - sim_start
        tqdm.write(f"    MCTS: All simulations completed in {sim_time:.2f}s")
        
        # Get action probabilities
        action_probs = np.zeros(self.env.num_bottles * self.env.num_bottles)
        for action, child in root.children.items():
            action_idx = self._action_to_idx(action)
            action_probs[action_idx] = child.visit_count
        
        # Apply temperature
        if np.sum(action_probs) > 0:
            action_probs = action_probs ** (1.0 / temperature)
            action_probs /= np.sum(action_probs)
        
        total_time = time.time() - search_start
        tqdm.write(f"    MCTS: Total search time {total_time:.2f}s")
        
        return action_probs, root.value()
    
    def _select_child(self, node):
        """Select child with highest UCB score"""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            # UCB formula
            ucb_score = child.value() + self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def _expand_and_evaluate(self, node):
        """Expand leaf node and get value estimate"""
        # Create temporary environment to simulate
        temp_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
        temp_env.bottles = node.parent.state.copy()
        
        # Apply action
        next_state, reward, done, _ = temp_env.step(node.action)
        node.state = next_state
        
        # Get neural network evaluation
        with torch.no_grad():
            policy, value = self.model(self.model.preprocess_state([next_state]).to(next(self.model.parameters()).device))
            value = value.item()
        
        # Expand node
        valid_moves = temp_env.get_valid_moves()
        valid_actions = [self._action_to_idx(move) for move in valid_moves]
        
        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
        policy_mask = np.zeros(len(policy))
        for action_idx in valid_actions:
            policy_mask[action_idx] = 1.0
        
        masked_policy = policy * policy_mask
        if np.sum(masked_policy) > 0:
            masked_policy /= np.sum(masked_policy)
        
        node.is_expanded = True
        for action_idx, prob in enumerate(masked_policy):
            if prob > 0:
                move = self._idx_to_action(action_idx)
                node.children[move] = Node(None, node, move, prob)
        
        return value
    
    def _backpropagate(self, search_path, value):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Alternate perspective for opponent
    
    def _action_to_idx(self, action):
        """Convert (from, to) action to linear index"""
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx
    
    def _idx_to_action(self, idx):
        """Convert linear index to (from, to) action"""
        from_idx = idx // self.env.num_bottles
        to_idx = idx % self.env.num_bottles
        return (from_idx, to_idx)

# =============================================================================
# 4. TRAINING PIPELINE - ALPHAZERO STYLE
# =============================================================================

class AlphaZeroTrainer:
    def __init__(self, env, model, lr=0.001, num_simulations=100, 
                 num_self_play_games=100, num_epochs=10, batch_size=32):
        self.env = env
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.mcts = MCTS(env, model, num_simulations)
        
        self.num_simulations = num_simulations
        self.num_self_play_games = num_self_play_games
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.replay_buffer = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def self_play(self, temperature=1.0):
        """Generate self-play data"""
        games_data = []
        total_moves = 0
        total_wins = 0
        
        tqdm.write("Starting self-play games...")
        
        # Progress bar cho self-play games
        game_pbar = tqdm(range(self.num_self_play_games), 
                        desc="🎮 Self-Play", 
                        bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%',
                        leave=True,
                        position=0)
        
        for game_idx in game_pbar:
            game_start_time = time.time()
            state = self.env.reset()
            game_history = []
            move_count = 0
            max_moves = 100  # Prevent infinite games
            
            tqdm.write(f"\n--- Game {game_idx + 1}/{self.num_self_play_games} ---")
            
            while move_count < max_moves:
                move_start = time.time()
                
                # Get action probabilities from MCTS
                tqdm.write(f"  Move {move_count + 1}: Starting MCTS search...")
                mcts_start = time.time()
                action_probs, value = self.mcts.search(state, temperature)
                mcts_time = time.time() - mcts_start
                tqdm.write(f"  Move {move_count + 1}: MCTS completed in {mcts_time:.2f}s, value={value:.3f}")
                
                # Store training data
                game_history.append((state, action_probs))
                
                # Sample action with validation
                if np.sum(action_probs) > 0:
                    action_probs_normalized = action_probs / np.sum(action_probs)
                else:
                    action_probs_normalized = action_probs
                action_idx = np.random.choice(len(action_probs_normalized), p=action_probs_normalized)
                action = self.mcts._idx_to_action(action_idx)
                tqdm.write(f"  Move {move_count + 1}: Selected action {action}")
                
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                move_count += 1
                move_time = time.time() - move_start
                tqdm.write(f"  Move {move_count}: Completed in {move_time:.2f}s, reward={reward:.2f}, done={done}")
                
                if done:
                    # Assign final values to all states
                    for i, (hist_state, hist_probs) in enumerate(game_history):
                        games_data.append((hist_state, hist_probs, reward))
                    
                    if reward > 0:
                        total_wins += 1
                        tqdm.write(f"  ✓ Game {game_idx + 1}: WON in {move_count} moves!")
                    else:
                        tqdm.write(f"  ✗ Game {game_idx + 1}: LOST after {move_count} moves")
                    
                    total_moves += move_count
                    
                    # Update description with stats
                    win_rate = total_wins / (game_idx + 1)
                    avg_moves = total_moves / (game_idx + 1)
                    game_pbar.set_description(
                        f"🎮 Self-Play (Win: {win_rate:.1%}, Moves: {avg_moves:.1f})"
                    )
                    break
                
                state = next_state
            
            if move_count >= max_moves:
                tqdm.write(f"  ⚠ Game {game_idx + 1}: TIMEOUT after {max_moves} moves")
                # Still save the data even if timeout
                for i, (hist_state, hist_probs) in enumerate(game_history):
                    games_data.append((hist_state, hist_probs, -1.0))  # Penalty for timeout
            
            game_time = time.time() - game_start_time
            tqdm.write(f"Game {game_idx + 1} total time: {game_time:.2f}s\n")
            
            # Update progress bar
            game_pbar.update(1)
        
        game_pbar.close()
        
        self.replay_buffer.extend(games_data)
        # Keep only recent games
        if len(self.replay_buffer) > 10000:
            self.replay_buffer = self.replay_buffer[-10000:]
        
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"Self-play completed: {len(games_data)} samples | "
                  f"Win rate: {total_wins/self.num_self_play_games:.1%} | "
                  f"Avg moves: {total_moves/self.num_self_play_games:.1f}")
        tqdm.write(f"{'='*70}\n")
        
        return games_data
    
    def train_epoch(self):
        """Train for one epoch on replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return float('inf'), 0, 0
        
        self.model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        # Shuffle data
        random.shuffle(self.replay_buffer)
        
        # Progress bar cho training batches
        batch_indices = list(range(0, len(self.replay_buffer), self.batch_size))
        batch_pbar = tqdm(batch_indices, 
                         desc="   📊 Training",
                         bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%',
                         leave=False,
                         position=1)
        
        for i in batch_pbar:
            batch = self.replay_buffer[i:i + self.batch_size]
            if len(batch) < self.batch_size:
                continue
            
            states = [item[0] for item in batch]
            target_policies = [item[1] for item in batch]
            target_values = [item[2] for item in batch]
            
            # Convert to tensors
            state_batch = self.model.preprocess_state(states).to(self.device)
            target_policy_batch = torch.FloatTensor(target_policies).to(self.device)
            target_value_batch = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(state_batch)
            
            # Compute losses
            # Use log softmax + KL divergence for probability matching
            log_policy = F.log_softmax(policy_pred, dim=1)
            policy_loss = -torch.sum(target_policy_batch * log_policy) / len(batch)
            value_loss = F.mse_loss(value_pred, target_value_batch)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            # Update description with current loss
            batch_pbar.set_description(f"   📊 Training (Loss: {loss.item():.4f})")
        
        batch_pbar.close()
        
        if num_batches > 0:
            return total_loss / num_batches, total_policy_loss / num_batches, total_value_loss / num_batches
        return float('inf'), 0, 0
    
    def train(self, total_iterations=100, save_path='watersort_model.pth'):
        """Main training loop"""
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        best_loss = float('inf')
        training_start = time.time()
        
        for iteration in range(total_iterations):
            iter_start = time.time()
            
            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"📍 ITERATION {iteration + 1}/{total_iterations}")
            tqdm.write(f"{'='*70}\n")
            
            # Self-play phase
            tqdm.write(f"[1/3] Self-Play Phase")
            self_play_start = time.time()
            self_play_data = self.self_play(temperature=1.0)
            self_play_time = time.time() - self_play_start
            
            # Training phase
            tqdm.write(f"\n[2/3] Training Phase ({self.num_epochs} epochs)")
            train_start = time.time()
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            
            for epoch in range(self.num_epochs):
                loss, policy_loss, value_loss = self.train_epoch()
                epoch_losses.append(loss)
                epoch_policy_losses.append(policy_loss)
                epoch_value_losses.append(value_loss)
            
            train_time = time.time() - train_start
            avg_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)
            
            tqdm.write(f"Training completed: Loss={avg_loss:.4f}, "
                      f"Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")
            
            # Save model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                    'iteration': iteration
                }, save_path)
                tqdm.write(f"✅ Model saved! Best loss: {best_loss:.4f}")
            
            # Evaluate
            tqdm.write(f"\n[3/3] Evaluation")
            if iteration % 5 == 0:
                self.evaluate_model()
            else:
                tqdm.write("Skipped (runs every 5 iterations)")
            
            iter_time = time.time() - iter_start
            eta_minutes = iter_time * (total_iterations - iteration - 1) / 60
            tqdm.write(f"\n✓ Iteration completed in {iter_time:.1f}s | ETA: {eta_minutes:.1f}min\n")
    
    def evaluate_model(self, num_games=10):
        """Evaluate current model performance"""
        wins = 0
        total_moves = 0
        
        eval_pbar = tqdm(range(num_games), 
                        desc="   🎯 Evaluating",
                        bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}%',
                        leave=False,
                        position=1)
        
        for game_idx in eval_pbar:
            state = self.env.reset()
            moves = 0
            max_moves = 100
            
            while moves < max_moves:
                action_probs, _ = self.mcts.search(state, temperature=1.0)
                action_idx = np.argmax(action_probs)
                action = self.mcts._idx_to_action(action_idx)
                
                next_state, reward, done, _ = self.env.step(action)
                moves += 1
                
                if done:
                    if reward > 0:
                        wins += 1
                    break
                
                state = next_state
            
            total_moves += moves
            
            # Update description with current stats
            win_rate = wins / (game_idx + 1)
            avg_moves = total_moves / (game_idx + 1)
            eval_pbar.set_description(f"   🎯 Eval (Win: {win_rate:.1%}, Moves: {avg_moves:.1f})")
        
        eval_pbar.close()
        
        win_rate = wins / num_games
        avg_moves = total_moves / num_games
        
        tqdm.write(f"Evaluation: Win rate {win_rate:.1%} ({wins}/{num_games}), "
                  f"Avg moves {avg_moves:.1f}")
        
        return win_rate, avg_moves

# =============================================================================
# 5. FINE-TUNING & INFERENCE
# =============================================================================

def load_model_for_finetuning(model_path='watersort_model.pth', env=None, lr=0.0001):
    """Load pre-trained model for fine-tuning"""
    if env is None:
        env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully from {model_path}")
            
            # Create trainer with lower learning rate for fine-tuning
            trainer = AlphaZeroTrainer(env, model, lr=lr)
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            return trainer
        else:
            print(f"Model file not found: {model_path}")
            print("Creating new model...")
            return AlphaZeroTrainer(env, model, lr=lr)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
        return AlphaZeroTrainer(env, model, lr=lr)

def solve_puzzle(initial_state, model_path='watersort_model.pth', env=None, num_simulations=200):
    """Use trained model to solve a specific puzzle"""
    if env is None:
        env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded for inference from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
            print("Using untrained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
    
    mcts = MCTS(env, model, num_simulations)
    state = initial_state
    solution = []
    
    print("Initial state:")
    env.bottles = state
    env.render()
    
    max_moves = 50
    for move_count in range(max_moves):
        action_probs, value = mcts.search(state, temperature=0.0)
        action_idx = np.argmax(action_probs)
        action = mcts._idx_to_action(action_idx)
        
        next_state, reward, done, _ = env.step(action)
        solution.append(action)
        
        print(f"Move {move_count + 1}: {action}")
        env.render()
        
        if done:
            print(f"Puzzle solved in {move_count + 1} moves!")
            return solution
        
        state = next_state
    
    print(f"Failed to solve in {max_moves} moves")
    return solution

# =============================================================================
# 6. MAIN EXECUTION - READY FOR COLAB
# =============================================================================

def main():
    """Main function to run training"""
    print("🚀 WaterSort AI - AlphaZero Implementation")
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Initialize environment and model
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
    
    # Create trainer - REDUCED PARAMETERS FOR DEBUGGING
    print("\n⚠️  DEBUG MODE: Using reduced parameters for faster testing")
    trainer = AlphaZeroTrainer(
        env=env,
        model=model,
        lr=0.001,
        num_simulations=50,  # Reduced from 100
        num_self_play_games=5,  # Reduced from 50 for testing
        num_epochs=3,  # Reduced from 10
        batch_size=32
    )
    
    print(f"Config: {trainer.num_simulations} MCTS sims, {trainer.num_self_play_games} games, {trainer.num_epochs} epochs\n")
    
    # Start training
    trainer.train(total_iterations=2)  # Just 2 iterations for testing

def demo_finetuning():
    """Demo fine-tuning on pre-trained model"""
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    
    # Try to load pre-trained model
    trainer = load_model_for_finetuning(
        model_path='watersort_model.pth',
        env=env,
        lr=0.0001  # Lower LR for fine-tuning
    )
    
    # Continue training
    trainer.train(total_iterations=20)

def demo_solving():
    """Demo using model to solve puzzles"""
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    initial_state = env.reset()
    
    solution = solve_puzzle(
        initial_state=initial_state,
        model_path='/content/drive/MyDrive/watersort_model.pth',
        env=env
    )
    
    return solution

# =============================================================================
# RUN TRAINING
# =============================================================================

if __name__ == "__main__":
    # Uncomment one of these to run:
    
    # 1. Full training from scratch
    main()
    
    # 2. Fine-tuning existing model
    # demo_finetuning()
    
    # 3. Solve specific puzzle
    # solution = demo_solving()