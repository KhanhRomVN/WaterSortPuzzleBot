# =============================================================================
# WATER SORT PUZZLE AI - MODEL TESTING
# Test pre-trained model on new puzzles
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import List, Tuple, Optional
import time
import os
from tqdm import tqdm

# =============================================================================
# 1. WATER SORT ENVIRONMENT (Same as training)
# =============================================================================

class WaterSortEnv:
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.num_colors = num_colors
        self.bottle_height = bottle_height
        self.num_bottles = num_bottles
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        colors = list(range(1, self.num_colors + 1)) * (self.bottle_height - 1)
        random.shuffle(colors)
        
        self.bottles = np.zeros((self.num_bottles, self.bottle_height), dtype=int)
        color_idx = 0
        
        for i in range(self.num_colors):
            for j in range(self.bottle_height - 1):
                if color_idx < len(colors):
                    for bottle_idx in range(self.num_bottles):
                        if np.sum(self.bottles[bottle_idx] > 0) < self.bottle_height - 1:
                            empty_pos = np.where(self.bottles[bottle_idx] == 0)[0][-1]
                            self.bottles[bottle_idx, empty_pos] = colors[color_idx]
                            color_idx += 1
                            break
        
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
        
        if np.sum(from_bottle > 0) == 0:
            return False
        
        if np.sum(to_bottle > 0) == self.bottle_height:
            return False
        
        source_top_idx = np.where(from_bottle > 0)[0]
        if len(source_top_idx) == 0:
            return False
        source_top_color = from_bottle[source_top_idx[0]]
        
        dest_top_idx = np.where(to_bottle > 0)[0]
        if len(dest_top_idx) == 0:
            return True
        dest_top_color = to_bottle[dest_top_idx[0]]
        
        return source_top_color == dest_top_color
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute move and return new state, reward, done, info"""
        from_idx, to_idx = action
        
        if not self._is_valid_move(from_idx, to_idx):
            return self.get_state(), -1, False, {"error": "Invalid move"}
        
        self._pour_liquid(from_idx, to_idx)
        done = self.is_solved()
        reward = 10.0 if done else -0.1
        
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
        
        pour_amount = 1
        for i in range(source_top_idx + 1, len(from_bottle)):
            if from_bottle[i] == source_color:
                pour_amount += 1
            else:
                break
        
        dest_empty = np.where(to_bottle == 0)[0]
        if len(dest_empty) == 0:
            return
        
        available_space = len(dest_empty)
        actual_pour = min(pour_amount, available_space)
        
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
        print("\nCurrent state:")
        for i, bottle in enumerate(self.bottles):
            bottle_str = ' '.join([str(int(x)) for x in bottle])
            print(f"  Bottle {i}: [{bottle_str}]")

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
        
        self.conv_input = nn.Conv2d(num_colors + 1, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        self.conv_policy = nn.Conv2d(channels, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * bottle_height * num_bottles, num_bottles * num_bottles)
        
        self.conv_value = nn.Conv2d(channels, 32, 1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * bottle_height * num_bottles, 256)
        self.fc_value2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)
        
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))
        
        return policy, value
    
    def preprocess_state(self, state_batch):
        """Convert state batch to neural network input format"""
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
# 3. SIMPLE GREEDY SOLVER (No MCTS)
# =============================================================================

class GreedySolver:
    """Simple greedy solver for testing without heavy MCTS"""
    
    def __init__(self, env, model, device='cpu'):
        self.env = env
        self.model = model
        self.device = device
    
    def solve(self, max_moves=50):
        """Solve puzzle using greedy policy from model"""
        state = self.env.get_state()
        moves = 0
        solution = []
        
        while moves < max_moves:
            if self.env.is_solved():
                return solution, True, moves
            
            valid_moves = self.env.get_valid_moves()
            if not valid_moves:
                return solution, False, moves
            
            # Get model prediction
            with torch.no_grad():
                state_tensor = self.model.preprocess_state([state]).to(self.device)
                policy, value = self.model(state_tensor)
                policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
            
            # Mask invalid moves
            valid_indices = [self._action_to_idx(m) for m in valid_moves]
            masked_policy = np.zeros_like(policy)
            for idx in valid_indices:
                masked_policy[idx] = policy[idx]
            
            if np.sum(masked_policy) == 0:
                masked_policy = np.ones(len(policy)) / len(policy)
            
            # Select best move
            action_idx = np.argmax(masked_policy)
            action = self._idx_to_action(action_idx)
            
            next_state, reward, done, _ = self.env.step(action)
            solution.append(action)
            moves += 1
            
            if done:
                return solution, True, moves
            
            state = next_state
        
        return solution, False, moves
    
    def _action_to_idx(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx
    
    def _idx_to_action(self, idx):
        from_idx = idx // self.env.num_bottles
        to_idx = idx % self.env.num_bottles
        return (from_idx, to_idx)

# =============================================================================
# 4. MAIN TESTING FUNCTION
# =============================================================================

def test_model():
    """Main testing function"""
    
    print("="*70)
    print("  WATER SORT PUZZLE AI - MODEL TESTING")
    print("="*70)
    
    # Get model path from user
    print("\nAvailable files in current directory:")
    files = [f for f in os.listdir('.') if f.endswith('.pth')]
    if files:
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")
    else:
        print("  No .pth files found!")
    
    model_path = input("\nEnter model file name (or path): ").strip()
    
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found!")
        return
    
    print(f"\nLoading model from: {model_path}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize environment and model
    env = WaterSortEnv(num_colors=6, bottle_height=4, num_bottles=8)
    model = WaterSortNet(env.num_bottles, env.bottle_height, env.num_colors)
    
    # Load model
    file_size = os.path.getsize(model_path)
    if file_size < 1000:
        print(f"✗ Error: File quá nhỏ ({file_size} bytes). File có thể bị hỏng!")
        return
    
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    
    model_loaded = False
    checkpoint = None
    
    # Cách 1: Thử load bình thường
    try:
        print("  [DEBUG] Cách 1: Thử torch.load với weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_loaded = True
        print("  ✓ Cách 1 thành công!")
    except RuntimeError as e:
        print(f"  [DEBUG] Cách 1 failed: {str(e)[:100]}")
        
        # Cách 2: Thử dùng pickle trực tiếp
        try:
            print("  [DEBUG] Cách 2: Thử load file với pickle...")
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            model_loaded = True
            print("  ✓ Cách 2 thành công!")
        except Exception as e2:
            print(f"  [DEBUG] Cách 2 failed: {str(e2)[:100]}")
            
            # Cách 3: Thử load state_dict trực tiếp (nếu file là state_dict, không phải checkpoint)
            try:
                print("  [DEBUG] Cách 3: File có thể là state_dict, thử load trực tiếp...")
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                print("  ✓ Cách 3 thành công! (File là state_dict, không phải checkpoint)")
                model_loaded = True
            except Exception as e3:
                print(f"  [DEBUG] Cách 3 failed: {str(e3)[:100]}")
                print(f"\n✗ Error: Không thể load file '{model_path}'")
                print(f"  Tất cả phương pháp đều thất bại:")
                print(f"    - Cách 1 (torch.load): {str(e)[:80]}")
                print(f"    - Cách 2 (pickle): {str(e2)[:80]}")
                print(f"    - Cách 3 (state_dict): {str(e3)[:80]}")
                print(f"\n  Gợi ý:")
                print(f"    1. Thử file khác: watersort_warmup.pth")
                print(f"    2. Tạo model test mới bằng script create_dummy_model.py")
                print(f"    3. Kiểm tra file có bị corruption không")
                return
    
    # Nếu load thành công qua checkpoint
    if model_loaded and checkpoint is not None:
        try:
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    print("  [DEBUG] Checkpoint chứa 'model_state_dict', loading...")
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    print("  [DEBUG] Checkpoint chứa 'state_dict', loading...")
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    print(f"  [DEBUG] Checkpoint keys: {list(checkpoint.keys())}")
                    print("  ✗ Error: Checkpoint không chứa 'model_state_dict' hoặc 'state_dict'")
                    return
            
            model.to(device)
            model.eval()
            print(f"✓ Model loaded successfully!")
            if isinstance(checkpoint, dict) and 'loss' in checkpoint:
                print(f"  Training loss: {checkpoint['loss']:.4f}")
            if isinstance(checkpoint, dict) and 'iteration' in checkpoint:
                print(f"  Iteration: {checkpoint['iteration']}")
        except Exception as e:
            print(f"✗ Error loading state_dict: {e}")
            return
    
    # Test parameters
    num_tests = int(input("\nHow many puzzles to test? (default: 10): ").strip() or "10")
    
    print(f"\n{'='*70}")
    print(f"Testing model on {num_tests} puzzles...")
    print(f"{'='*70}\n")
    
    # Create solver
    solver = GreedySolver(env, model, device)
    
    # Run tests
    wins = 0
    total_moves = 0
    all_results = []
    
    pbar = tqdm(range(num_tests), desc="Testing puzzles")
    
    for test_idx in pbar:
        env.reset()
        solution, success, moves = solver.solve(max_moves=100)
        
        if success:
            wins += 1
            total_moves += moves
            all_results.append((True, moves))
        else:
            all_results.append((False, moves))
        
        win_rate = wins / (test_idx + 1)
        avg_moves = total_moves / max(wins, 1)
        pbar.set_description(f"Testing (Win: {win_rate:.1%}, Avg: {avg_moves:.1f})")
    
    # Print results
    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}\n")
    print(f"Success Rate: {wins}/{num_tests} ({100*wins/num_tests:.1f}%)")
    if wins > 0:
        print(f"Average Moves (solved): {total_moves/wins:.1f}")
    
    print(f"\nDetailed results:")
    for i, (success, moves) in enumerate(all_results, 1):
        status = "✓ SOLVED" if success else "✗ FAILED"
        print(f"  Puzzle {i:2d}: {status} ({moves} moves)")
    
    # Option to test a single puzzle manually
    print(f"\n{'='*70}")
    test_one = input("\nTest a single puzzle manually? (y/n): ").strip().lower()
    
    if test_one == 'y':
        print("\nGenerating new puzzle...")
        env.reset()
        env.render()
        
        print("\nSolving...")
        solution, success, moves = solver.solve(max_moves=100)
        
        if success:
            print(f"\n✓ Puzzle solved in {moves} moves!")
            print(f"Solution: {solution}")
            
            # Replay solution
            print("\nReplaying solution:")
            env.reset()
            for i, (from_idx, to_idx) in enumerate(solution, 1):
                print(f"\nMove {i}: Pour from bottle {from_idx} to bottle {to_idx}")
                env.step((from_idx, to_idx))
                env.render()
            
            print("\n✓ Final state - Puzzle solved!")
        else:
            print(f"\n✗ Failed to solve (tried {moves} moves)")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    test_model()