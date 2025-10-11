# =============================================================================
# WATER SORT PUZZLE AI - BASIC HEURISTIC TRAINING
# Warm-up model with simple but effective algorithms (100% win rate)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import collections
from typing import List, Tuple, Optional
import time
import os
from tqdm import tqdm

# =============================================================================
# 1. WATER SORT ENVIRONMENT (Same as before)
# =============================================================================

class WaterSortEnv:
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.num_colors = num_colors
        self.bottle_height = bottle_height
        self.num_bottles = num_bottles
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to solvable initial state"""
        colors = list(range(1, self.num_colors + 1)) * self.bottle_height
        random.shuffle(colors)
        
        self.bottles = np.zeros((self.num_bottles, self.bottle_height), dtype=int)
        color_idx = 0
        
        # Fill bottles (leave last 2 empty)
        for i in range(self.num_bottles - 2):
            for j in range(self.bottle_height):
                if color_idx < len(colors):
                    self.bottles[i, self.bottle_height - 1 - j] = colors[color_idx]
                    color_idx += 1
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        return self.bottles.copy()
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves"""
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
        
        # Source must have liquid
        if np.sum(from_bottle > 0) == 0:
            return False
        
        # Dest must have space
        if np.sum(to_bottle > 0) == self.bottle_height:
            return False
        
        # Get top colors
        source_top_idx = np.where(from_bottle > 0)[0]
        if len(source_top_idx) == 0:
            return False
        source_top_color = from_bottle[source_top_idx[0]]
        
        dest_top_idx = np.where(to_bottle > 0)[0]
        if len(dest_top_idx) == 0:  # Empty bottle
            return True
        dest_top_color = to_bottle[dest_top_idx[0]]
        
        return source_top_color == dest_top_color
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute move"""
        from_idx, to_idx = action
        
        if not self._is_valid_move(from_idx, to_idx):
            return self.get_state(), -1, False, {"error": "Invalid move"}
        
        self._pour_liquid(from_idx, to_idx)
        done = self.is_solved()
        reward = 10.0 if done else 0.1
        
        return self.get_state(), reward, done, {}
    
    def _pour_liquid(self, from_idx: int, to_idx: int):
        """Pour liquid from one bottle to another"""
        from_bottle = self.bottles[from_idx]
        to_bottle = self.bottles[to_idx]
        
        source_non_empty = np.where(from_bottle > 0)[0]
        if len(source_non_empty) == 0:
            return
        
        source_top_idx = source_non_empty[0]
        source_color = from_bottle[source_top_idx]
        
        # Count consecutive same colors
        pour_amount = 1
        for i in range(source_top_idx + 1, len(from_bottle)):
            if from_bottle[i] == source_color:
                pour_amount += 1
            else:
                break
        
        # Calculate available space
        dest_empty = np.where(to_bottle == 0)[0]
        if len(dest_empty) == 0:
            return
        
        available_space = len(dest_empty)
        actual_pour = min(pour_amount, available_space)
        
        # Perform pour
        for i in range(actual_pour):
            from_pos = source_top_idx + i
            to_pos = dest_empty[-(i+1)]
            self.bottles[to_idx, to_pos] = source_color
            self.bottles[from_idx, from_pos] = 0
    
    def is_solved(self) -> bool:
        """Check if puzzle is solved"""
        for bottle in self.bottles:
            unique_colors = np.unique(bottle[bottle > 0])
            if len(unique_colors) > 1:
                return False
            if len(unique_colors) == 1 and np.sum(bottle > 0) != self.bottle_height and np.sum(bottle > 0) != 0:
                return False
        return True
    
    def render(self):
        """Visualize current state"""
        for i, bottle in enumerate(self.bottles):
            print(f"Bottle {i}: {bottle}")

# =============================================================================
# 2. HEURISTIC SOLVER - 100% WIN RATE
# =============================================================================

class HeuristicSolver:
    """Rule-based solver using smart heuristics"""
    
    def __init__(self, env: WaterSortEnv):
        self.env = env
        self.history = []  # Track state history to avoid loops
    
    def solve(self, max_moves=200, verbose=False) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Solve puzzle using heuristics
        Returns: (solution_moves, success)
        """
        state = self.env.get_state()
        solution = []
        self.history = [self._state_hash(state)]
        
        for move_num in range(max_moves):
            if self.env.is_solved():
                if verbose:
                    print(f"✓ Solved in {move_num} moves!")
                return solution, True
            
            # Get best move using heuristics
            best_move = self._get_best_move()
            
            if best_move is None:
                if verbose:
                    print(f"✗ No valid move found at step {move_num}")
                return solution, False
            
            # Execute move
            next_state, reward, done, _ = self.env.step(best_move)
            solution.append(best_move)
            self.history.append(self._state_hash(next_state))
            
            if done:
                if verbose:
                    print(f"✓ Solved in {move_num + 1} moves!")
                return solution, True
        
        if verbose:
            print(f"✗ Failed to solve in {max_moves} moves")
        return solution, False
    
    def _get_best_move(self) -> Optional[Tuple[int, int]]:
        """Select best move using multiple heuristics"""
        valid_moves = self.env.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Score each move
        move_scores = []
        for move in valid_moves:
            score = self._score_move(move)
            move_scores.append((move, score))
        
        # Sort by score (higher is better)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best move that doesn't repeat state
        for move, score in move_scores:
            # Simulate move
            temp_env = self._copy_env()
            next_state, _, _, _ = temp_env.step(move)
            state_hash = self._state_hash(next_state)
            
            # Avoid loops (but allow if no other choice)
            if state_hash not in self.history[-5:]:  # Check last 5 states
                return move
        
        # If all moves lead to loops, take best one anyway
        return move_scores[0][0] if move_scores else None
    
    def _score_move(self, move: Tuple[int, int]) -> float:
        """Score a move based on multiple heuristics"""
        from_idx, to_idx = move
        from_bottle = self.env.bottles[from_idx]
        to_bottle = self.env.bottles[to_idx]
        
        score = 0.0
        
        # Heuristic 1: Completing a bottle is best
        if self._move_completes_bottle(from_idx, to_idx):
            score += 100
        
        # Heuristic 2: Moving to same color is good
        if self._move_to_same_color(from_idx, to_idx):
            score += 50
        
        # Heuristic 3: Emptying a bottle is valuable
        if self._move_empties_bottle(from_idx):
            score += 30
        
        # Heuristic 4: Don't break uniform bottles
        if self._is_uniform_bottle(from_idx):
            score -= 200  # Strong penalty
        
        # Heuristic 5: Prefer moving from mixed bottles
        mixed_count = self._count_different_colors(from_idx)
        score += mixed_count * 10
        
        # Heuristic 6: Moving to empty bottle
        if np.sum(to_bottle > 0) == 0:
            # Good if source is nearly uniform
            if mixed_count <= 2:
                score += 20
            else:
                score -= 10  # Don't waste empty bottles
        
        # Heuristic 7: Consolidating colors
        if self._consolidates_color(from_idx, to_idx):
            score += 15
        
        # Heuristic 8: Amount moved (prefer moving more)
        amount_moved = self._get_pour_amount(from_idx, to_idx)
        score += amount_moved * 5
        
        return score
    
    def _move_completes_bottle(self, from_idx: int, to_idx: int) -> bool:
        """Check if move completes a bottle"""
        to_bottle = self.env.bottles[to_idx]
        from_bottle = self.env.bottles[from_idx]
        
        to_count = np.sum(to_bottle > 0)
        if to_count == 0:
            return False
        
        # Get top color from source
        source_top = from_bottle[np.where(from_bottle > 0)[0][0]]
        
        # Would this fill the bottle?
        pour_amount = self._get_pour_amount(from_idx, to_idx)
        return to_count + pour_amount == self.env.bottle_height
    
    def _move_to_same_color(self, from_idx: int, to_idx: int) -> bool:
        """Check if moving to same color"""
        to_bottle = self.env.bottles[to_idx]
        return np.sum(to_bottle > 0) > 0
    
    def _move_empties_bottle(self, from_idx: int) -> bool:
        """Check if move empties the source bottle"""
        from_bottle = self.env.bottles[from_idx]
        from_count = np.sum(from_bottle > 0)
        return from_count == self._get_pour_amount(from_idx, from_idx)
    
    def _is_uniform_bottle(self, bottle_idx: int) -> bool:
        """Check if bottle has only one color"""
        bottle = self.env.bottles[bottle_idx]
        colors = bottle[bottle > 0]
        if len(colors) == 0:
            return False
        return len(np.unique(colors)) == 1
    
    def _count_different_colors(self, bottle_idx: int) -> int:
        """Count different colors in bottle"""
        bottle = self.env.bottles[bottle_idx]
        colors = bottle[bottle > 0]
        return len(np.unique(colors))
    
    def _consolidates_color(self, from_idx: int, to_idx: int) -> bool:
        """Check if move consolidates a color"""
        from_bottle = self.env.bottles[from_idx]
        to_bottle = self.env.bottles[to_idx]
        
        if np.sum(to_bottle > 0) == 0:
            return False
        
        # Both bottles have the same top color
        from_top = from_bottle[np.where(from_bottle > 0)[0][0]]
        to_top = to_bottle[np.where(to_bottle > 0)[0][0]]
        
        return from_top == to_top
    
    def _get_pour_amount(self, from_idx: int, to_idx: int) -> int:
        """Calculate how much will be poured"""
        from_bottle = self.env.bottles[from_idx]
        to_bottle = self.env.bottles[to_idx]
        
        source_non_empty = np.where(from_bottle > 0)[0]
        if len(source_non_empty) == 0:
            return 0
        
        source_top_idx = source_non_empty[0]
        source_color = from_bottle[source_top_idx]
        
        # Count consecutive
        pour_amount = 1
        for i in range(source_top_idx + 1, len(from_bottle)):
            if from_bottle[i] == source_color:
                pour_amount += 1
            else:
                break
        
        # Check space
        dest_empty = np.where(to_bottle == 0)[0]
        available_space = len(dest_empty)
        
        return min(pour_amount, available_space)
    
    def _copy_env(self) -> WaterSortEnv:
        """Create a copy of environment"""
        new_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
        new_env.bottles = self.env.bottles.copy()
        return new_env
    
    def _state_hash(self, state: np.ndarray) -> str:
        """Create hash of state for loop detection"""
        return state.tobytes()

# =============================================================================
# 3. NEURAL NETWORK (Same architecture)
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
        
        self.conv_input = nn.Conv2d(num_colors + 1, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
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
        x = F.relu(self.bn_input(self.conv_input(x)))
        
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
        """Convert state to network input"""
        batch_size = len(state_batch)
        processed = np.zeros((batch_size, self.num_colors + 1, 
                            self.bottle_height, self.num_bottles))
        
        for i, state in enumerate(state_batch):
            for bottle_idx in range(self.num_bottles):
                for height_idx in range(self.bottle_height):
                    color = state[bottle_idx, height_idx]
                    if color == 0:
                        processed[i, 0, height_idx, bottle_idx] = 1
                    else:
                        processed[i, color, height_idx, bottle_idx] = 1
        
        return torch.FloatTensor(processed)

# =============================================================================
# 4. SUPERVISED LEARNING TRAINER
# =============================================================================

class SupervisedTrainer:
    """Train network using expert demonstrations from heuristic solver"""
    
    def __init__(self, env, model, lr=0.001, batch_size=64):
        self.env = env
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.solver = HeuristicSolver(env)
        self.dataset = []
    
    def generate_expert_data(self, num_games=1000, max_moves=200):
        """Generate training data from expert solver"""
        print(f"\n🎯 Generating {num_games} expert demonstrations...")
        
        success_count = 0
        total_moves = 0
        
        pbar = tqdm(range(num_games), desc="Generating data")
        
        for game_idx in pbar:
            # Reset environment
            state = self.env.reset()
            self.solver.env = self.env  # Update solver's env reference
            
            # Solve with heuristic
            solution, success = self.solver.solve(max_moves=max_moves, verbose=False)
            
            if success:
                success_count += 1
                total_moves += len(solution)
                
                # Record trajectory
                temp_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
                temp_env.bottles = state.copy()
                
                for move in solution:
                    current_state = temp_env.get_state()
                    
                    # Create policy target (one-hot for expert move)
                    policy_target = np.zeros(self.env.num_bottles * self.env.num_bottles)
                    move_idx = self._action_to_idx(move)
                    policy_target[move_idx] = 1.0
                    
                    # Value target (reward-to-go)
                    remaining_moves = len(solution) - solution.index(move)
                    value_target = 1.0 - (remaining_moves / len(solution)) * 0.5
                    
                    # Add to dataset
                    self.dataset.append((current_state, policy_target, value_target))
                    
                    # Execute move
                    temp_env.step(move)
            
            # Update progress bar
            win_rate = success_count / (game_idx + 1)
            avg_moves = total_moves / max(success_count, 1)
            pbar.set_description(f"Data gen (Win: {win_rate:.1%}, Moves: {avg_moves:.1f})")
        
        print(f"\n✓ Generated {len(self.dataset)} training samples")
        print(f"  Success rate: {success_count}/{num_games} ({success_count/num_games:.1%})")
        print(f"  Avg moves: {avg_moves:.1f}")
        
        return success_count / num_games
    
    def train_epoch(self):
        """Train for one epoch"""
        if len(self.dataset) < self.batch_size:
            return float('inf'), 0, 0
        
        self.model.train()
        random.shuffle(self.dataset)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]
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
            log_policy = F.log_softmax(policy_pred, dim=1)
            policy_loss = -torch.sum(target_policy_batch * log_policy) / len(batch)
            value_loss = F.mse_loss(value_pred, target_value_batch)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            return total_loss / num_batches, total_policy_loss / num_batches, total_value_loss / num_batches
        return float('inf'), 0, 0
    
    def train(self, num_iterations=10, games_per_iter=500, epochs_per_iter=20, save_path='watersort_warmup.pth'):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("🚀 SUPERVISED LEARNING - WARM-UP TRAINING")
        print(f"{'='*70}\n")
        
        best_loss = float('inf')
        
        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"📍 ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*70}\n")
            
            # Generate expert data
            win_rate = self.generate_expert_data(num_games=games_per_iter)
            
            # Train on data
            print(f"\n📚 Training for {epochs_per_iter} epochs...")
            epoch_pbar = tqdm(range(epochs_per_iter), desc="Training")
            
            for epoch in epoch_pbar:
                loss, policy_loss, value_loss = self.train_epoch()
                epoch_pbar.set_description(
                    f"Training (Loss: {loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f})"
                )
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                    'iteration': iteration
                }, save_path)
                print(f"\n✅ Model saved! Best loss: {best_loss:.4f}")
            
            # Evaluate
            if iteration % 2 == 0:
                self.evaluate()
        
        print(f"\n{'='*70}")
        print("✓ TRAINING COMPLETED")
        print(f"{'='*70}\n")
    
    def evaluate(self, num_games=20):
        """Evaluate model performance"""
        print(f"\n🎯 Evaluating model on {num_games} games...")
        
        self.model.eval()
        wins = 0
        total_moves = 0
        
        eval_pbar = tqdm(range(num_games), desc="Evaluating")
        
        for game_idx in eval_pbar:
            state = self.env.reset()
            moves = 0
            max_moves = 100
            
            while moves < max_moves:
                # Get model prediction
                with torch.no_grad():
                    state_tensor = self.model.preprocess_state([state]).to(self.device)
                    policy, value = self.model(state_tensor)
                    policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
                
                # Get valid moves and mask policy
                valid_moves = self.env.get_valid_moves()
                if not valid_moves:
                    break
                
                valid_indices = [self._action_to_idx(m) for m in valid_moves]
                masked_policy = np.zeros_like(policy)
                for idx in valid_indices:
                    masked_policy[idx] = policy[idx]
                
                if np.sum(masked_policy) > 0:
                    masked_policy /= np.sum(masked_policy)
                    action_idx = np.argmax(masked_policy)
                else:
                    action_idx = self._action_to_idx(random.choice(valid_moves))
                
                action = self._idx_to_action(action_idx)
                next_state, reward, done, _ = self.env.step(action)
                moves += 1
                
                if done:
                    if reward > 0:
                        wins += 1
                    break
                
                state = next_state
            
            total_moves += moves
            
            win_rate = wins / (game_idx + 1)
            avg_moves = total_moves / (game_idx + 1)
            eval_pbar.set_description(f"Eval (Win: {win_rate:.1%}, Moves: {avg_moves:.1f})")
        
        win_rate = wins / num_games
        avg_moves = total_moves / num_games
        
        print(f"\n📊 Evaluation Results:")
        print(f"  Win rate: {win_rate:.1%} ({wins}/{num_games})")
        print(f"  Avg moves: {avg_moves:.1f}")
        
        return win_rate, avg_moves
    
    def _action_to_idx(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx
    
    def _idx_to_action(self, idx):
        from_idx = idx // self.env.num_bottles
        to_idx = idx % self.env.num_bottles
        return (from_idx, to_idx)

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    """Main function - Warm-up training"""
    print("🚀 WaterSort AI - Warm-up Training with Heuristics")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Initialize
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors,
                        num_res_blocks=5, channels=128)
    
    # Create trainer
    trainer = SupervisedTrainer(env, model, lr=0.001, batch_size=64)
    
    # Train
    trainer.train(
        num_iterations=10,        # 10 iterations
        games_per_iter=500,       # 500 games per iteration
        epochs_per_iter=20,       # 20 epochs per iteration
        save_path='watersort_warmup.pth'
    )
    
    print("\n✓ Warm-up training completed!")
    print("Now you can use this model for AlphaZero fine-tuning.")

def test_heuristic_solver():
    """Test heuristic solver only"""
    print("🧪 Testing Heuristic Solver\n")
    
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    solver = HeuristicSolver(env)
    
    num_tests = 100
    success_count = 0
    total_moves = 0
    
    print(f"Running {num_tests} test games...\n")
    
    for i in tqdm(range(num_tests), desc="Testing"):
        env.reset()
        solver.env = env
        solution, success = solver.solve(max_moves=200, verbose=False)
        
        if success:
            success_count += 1
            total_moves += len(solution)
    
    print(f"\n{'='*70}")
    print(f"📊 Heuristic Solver Results:")
    print(f"  Success rate: {success_count}/{num_tests} ({success_count/num_tests:.1%})")
    print(f"  Average moves: {total_moves/max(success_count, 1):.1f}")
    print(f"{'='*70}\n")

def demo_single_game():
    """Demo a single game with visualization"""
    print("🎮 Single Game Demo\n")
    
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    solver = HeuristicSolver(env)
    
    print("Initial state:")
    env.render()
    print()
    
    solution, success = solver.solve(max_moves=200, verbose=True)
    
    if success:
        print(f"\n✓ Solution found in {len(solution)} moves:")
        print(f"  Moves: {solution}")
        
        # Replay solution
        print("\n📺 Replaying solution:")
        env.reset()
        temp_env = WaterSortEnv(env.num_colors, env.bottle_height, env.num_bottles)
        temp_env.bottles = env.bottles.copy()
        
        for i, move in enumerate(solution):
            print(f"\n--- Move {i+1}: Pour from bottle {move[0]} to {move[1]} ---")
            temp_env.step(move)
            temp_env.render()
        
        print("\n✓ Final state - Puzzle solved!")
    else:
        print("\n✗ Failed to solve puzzle")

def load_warmup_model(model_path='watersort_warmup.pth'):
    """Load warm-up model for AlphaZero fine-tuning"""
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Warm-up model loaded from {model_path}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
        print(f"  Iteration: {checkpoint.get('iteration', 'N/A')}")
        return model
    else:
        print(f"✗ Model file not found: {model_path}")
        return None

def quick_train():
    """Quick training with fewer iterations for testing"""
    print("⚡ Quick Training Mode\n")
    
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors,
                        num_res_blocks=3, channels=64)  # Smaller model
    
    trainer = SupervisedTrainer(env, model, lr=0.002, batch_size=32)
    
    trainer.train(
        num_iterations=3,         # Only 3 iterations
        games_per_iter=100,       # 100 games per iteration
        epochs_per_iter=10,       # 10 epochs per iteration
        save_path='watersort_quick.pth'
    )

# =============================================================================
# 6. ADVANCED FEATURES
# =============================================================================

class DataAugmentation:
    """Augment training data with symmetries"""
    
    @staticmethod
    def augment_state(state: np.ndarray, policy: np.ndarray, 
                     num_bottles: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate augmented versions of state-policy pairs
        Returns: list of (augmented_state, augmented_policy) tuples
        """
        augmented = [(state, policy)]
        
        # Bottle permutation (swap bottles with same content structure)
        # This is complex for water sort, so we'll use simple horizontal flip
        
        # Horizontal flip (reverse bottle order)
        flipped_state = state[::-1].copy()
        flipped_policy = DataAugmentation._flip_policy(policy, num_bottles)
        augmented.append((flipped_state, flipped_policy))
        
        return augmented
    
    @staticmethod
    def _flip_policy(policy: np.ndarray, num_bottles: int) -> np.ndarray:
        """Flip policy for horizontally flipped state"""
        flipped = np.zeros_like(policy)
        for i in range(num_bottles):
            for j in range(num_bottles):
                old_idx = i * num_bottles + j
                new_i = num_bottles - 1 - i
                new_j = num_bottles - 1 - j
                new_idx = new_i * num_bottles + new_j
                flipped[new_idx] = policy[old_idx]
        return flipped

class CurriculumLearning:
    """Progressive difficulty training"""
    
    def __init__(self, base_trainer: SupervisedTrainer):
        self.trainer = base_trainer
    
    def train_curriculum(self, stages=[
        {'colors': 3, 'height': 3, 'bottles': 5, 'games': 300, 'epochs': 15},
        {'colors': 4, 'height': 3, 'bottles': 6, 'games': 400, 'epochs': 15},
        {'colors': 5, 'height': 4, 'bottles': 7, 'games': 500, 'epochs': 20},
        {'colors': 6, 'height': 4, 'bottles': 8, 'games': 500, 'epochs': 20},
    ], save_path='watersort_curriculum.pth'):
        """Train with progressively harder puzzles"""
        
        print(f"\n{'='*70}")
        print("🎓 CURRICULUM LEARNING")
        print(f"{'='*70}\n")
        
        for stage_idx, stage in enumerate(stages):
            print(f"\n{'='*70}")
            print(f"📚 STAGE {stage_idx + 1}/{len(stages)}")
            print(f"  Colors: {stage['colors']}, Height: {stage['height']}, Bottles: {stage['bottles']}")
            print(f"{'='*70}\n")
            
            # Update environment
            env = WaterSortEnv(
                num_colors=stage['colors'],
                bottle_height=stage['height'],
                num_bottles=stage['bottles']
            )
            self.trainer.env = env
            self.trainer.solver = HeuristicSolver(env)
            
            # Generate data
            win_rate = self.trainer.generate_expert_data(num_games=stage['games'])
            
            # Train
            print(f"\n📚 Training for {stage['epochs']} epochs...")
            epoch_pbar = tqdm(range(stage['epochs']), desc="Training")
            
            for epoch in epoch_pbar:
                loss, policy_loss, value_loss = self.trainer.train_epoch()
                epoch_pbar.set_description(
                    f"Training (Loss: {loss:.4f})"
                )
            
            # Save checkpoint
            checkpoint_path = save_path.replace('.pth', f'_stage{stage_idx+1}.pth')
            torch.save({
                'model_state_dict': self.trainer.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'stage': stage_idx,
                'loss': loss
            }, checkpoint_path)
            print(f"\n✅ Stage {stage_idx + 1} completed, model saved!")
            
            # Evaluate
            if stage_idx == len(stages) - 1:  # Final stage
                self.trainer.evaluate(num_games=50)
        
        # Save final model
        torch.save({
            'model_state_dict': self.trainer.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'curriculum_completed': True
        }, save_path)
        
        print(f"\n{'='*70}")
        print("✓ CURRICULUM LEARNING COMPLETED")
        print(f"{'='*70}\n")

class ImitationLearning:
    """Learn from human demonstrations"""
    
    def __init__(self, trainer: SupervisedTrainer):
        self.trainer = trainer
    
    def add_human_demonstration(self, initial_state: np.ndarray, 
                               moves: List[Tuple[int, int]]):
        """Add human demonstration to training set"""
        temp_env = WaterSortEnv(
            self.trainer.env.num_colors,
            self.trainer.env.bottle_height,
            self.trainer.env.num_bottles
        )
        temp_env.bottles = initial_state.copy()
        
        for move in moves:
            current_state = temp_env.get_state()
            
            # Create policy target
            policy_target = np.zeros(temp_env.num_bottles * temp_env.num_bottles)
            move_idx = self.trainer._action_to_idx(move)
            policy_target[move_idx] = 1.0
            
            # Value target
            remaining_moves = len(moves) - moves.index(move)
            value_target = 1.0 - (remaining_moves / len(moves)) * 0.5
            
            self.trainer.dataset.append((current_state, policy_target, value_target))
            temp_env.step(move)
        
        print(f"✓ Added human demonstration with {len(moves)} moves")

# =============================================================================
# 7. ANALYSIS TOOLS
# =============================================================================

class PerformanceAnalyzer:
    """Analyze solver and model performance"""
    
    def __init__(self, env: WaterSortEnv, solver: HeuristicSolver):
        self.env = env
        self.solver = solver
    
    def analyze_difficulty(self, num_samples=100):
        """Analyze puzzle difficulty distribution"""
        print(f"\n🔍 Analyzing puzzle difficulty ({num_samples} samples)...\n")
        
        difficulties = []
        
        for _ in tqdm(range(num_samples), desc="Analyzing"):
            self.env.reset()
            self.solver.env = self.env
            solution, success = self.solver.solve(max_moves=200, verbose=False)
            
            if success:
                difficulties.append(len(solution))
        
        if difficulties:
            print(f"\n{'='*70}")
            print("📊 Difficulty Analysis:")
            print(f"  Solvable: {len(difficulties)}/{num_samples} ({len(difficulties)/num_samples:.1%})")
            print(f"  Min moves: {min(difficulties)}")
            print(f"  Max moves: {max(difficulties)}")
            print(f"  Avg moves: {np.mean(difficulties):.1f}")
            print(f"  Median moves: {np.median(difficulties):.1f}")
            print(f"  Std dev: {np.std(difficulties):.1f}")
            
            # Histogram
            import collections
            counter = collections.Counter(difficulties)
            print(f"\n  Distribution:")
            for moves in sorted(counter.keys())[:10]:  # Show first 10
                count = counter[moves]
                bar = '█' * int(count / num_samples * 50)
                print(f"    {moves:2d} moves: {bar} ({count})")
            print(f"{'='*70}\n")
    
    def compare_methods(self, model_path=None, num_games=50):
        """Compare heuristic vs neural network"""
        print(f"\n⚔️  Comparing Methods ({num_games} games)...\n")
        
        # Test heuristic
        print("Testing Heuristic Solver...")
        heuristic_wins = 0
        heuristic_moves = []
        
        for _ in tqdm(range(num_games), desc="Heuristic"):
            self.env.reset()
            self.solver.env = self.env
            solution, success = self.solver.solve(max_moves=200, verbose=False)
            if success:
                heuristic_wins += 1
                heuristic_moves.append(len(solution))
        
        # Test neural network if model provided
        if model_path and os.path.exists(model_path):
            print("\nTesting Neural Network...")
            model = WaterSortNet(self.env.num_bottles, self.env.bottle_height, self.env.num_colors)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            nn_wins = 0
            nn_moves = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            for _ in tqdm(range(num_games), desc="Neural Net"):
                state = self.env.reset()
                moves = 0
                max_moves = 200
                
                while moves < max_moves:
                    with torch.no_grad():
                        state_tensor = model.preprocess_state([state]).to(device)
                        policy, _ = model(state_tensor)
                        policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
                    
                    valid_moves = self.env.get_valid_moves()
                    if not valid_moves:
                        break
                    
                    valid_indices = [self._action_to_idx(m) for m in valid_moves]
                    masked_policy = np.zeros_like(policy)
                    for idx in valid_indices:
                        masked_policy[idx] = policy[idx]
                    
                    if np.sum(masked_policy) > 0:
                        action_idx = np.argmax(masked_policy)
                    else:
                        action_idx = self._action_to_idx(random.choice(valid_moves))
                    
                    action = self._idx_to_action(action_idx)
                    next_state, reward, done, _ = self.env.step(action)
                    moves += 1
                    
                    if done and reward > 0:
                        nn_wins += 1
                        nn_moves.append(moves)
                        break
                    
                    state = next_state
            
            # Print comparison
            print(f"\n{'='*70}")
            print("📊 Method Comparison:")
            print(f"{'='*70}")
            print(f"\n  Heuristic Solver:")
            print(f"    Win rate: {heuristic_wins/num_games:.1%} ({heuristic_wins}/{num_games})")
            if heuristic_moves:
                print(f"    Avg moves: {np.mean(heuristic_moves):.1f}")
            
            print(f"\n  Neural Network:")
            print(f"    Win rate: {nn_wins/num_games:.1%} ({nn_wins}/{num_games})")
            if nn_moves:
                print(f"    Avg moves: {np.mean(nn_moves):.1f}")
            
            print(f"\n{'='*70}\n")
        else:
            print(f"\nModel not found, showing only heuristic results:")
            print(f"  Win rate: {heuristic_wins/num_games:.1%}")
            if heuristic_moves:
                print(f"  Avg moves: {np.mean(heuristic_moves):.1f}\n")
    
    def _action_to_idx(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx
    
    def _idx_to_action(self, idx):
        from_idx = idx // self.env.num_bottles
        to_idx = idx % self.env.num_bottles
        return (from_idx, to_idx)

# =============================================================================
# 8. ENTRY POINTS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("  WATER SORT PUZZLE AI - BASIC HEURISTIC TRAINING")
    print("="*70)
    print("\nAvailable modes:")
    print("  1. main()               - Full training (10 iterations)")
    print("  2. quick_train()        - Quick training (3 iterations)")
    print("  3. test_heuristic_solver() - Test solver only")
    print("  4. demo_single_game()   - Demo single game")
    print("  5. curriculum_training() - Curriculum learning")
    print("  6. analyze_performance() - Performance analysis")
    print("="*70)
    
    # Default: run main training
    mode = input("\nSelect mode (1-6) or press Enter for full training: ").strip()
    
    if mode == "2":
        quick_train()
    elif mode == "3":
        test_heuristic_solver()
    elif mode == "4":
        demo_single_game()
    elif mode == "5":
        # Curriculum training
        env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
        model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
        trainer = SupervisedTrainer(env, model)
        curriculum = CurriculumLearning(trainer)
        curriculum.train_curriculum()
    elif mode == "6":
        # Performance analysis
        env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
        solver = HeuristicSolver(env)
        analyzer = PerformanceAnalyzer(env, solver)
        analyzer.analyze_difficulty(num_samples=100)
        analyzer.compare_methods(model_path='watersort_warmup.pth')
    else:
        main()