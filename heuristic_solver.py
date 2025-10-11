# heuristic_solver.py
# Phase 1: Generate training data from heuristic solver
# =============================================================================

import numpy as np
import random
import pickle
import os
from tqdm import tqdm

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
    
    def get_valid_moves(self) -> list:
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
    
    def step(self, action):
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

# =============================================================================
# 2. HEURISTIC SOLVER
# =============================================================================

class HeuristicSolver:
    """Rule-based solver using smart heuristics"""
    
    def __init__(self, env: WaterSortEnv):
        self.env = env
        self.history = []
    
    def solve(self, max_moves=200):
        """Solve puzzle using heuristics"""
        state = self.env.get_state()
        solution = []
        self.history = [self._state_hash(state)]
        
        for move_num in range(max_moves):
            if self.env.is_solved():
                return solution, True
            
            best_move = self._get_best_move()
            
            if best_move is None:
                return solution, False
            
            next_state, _, done, _ = self.env.step(best_move)
            solution.append(best_move)
            self.history.append(self._state_hash(next_state))
            
            if done:
                return solution, True
        
        return solution, False
    
    def _get_best_move(self):
        """Select best move using heuristics"""
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
            temp_env = self._copy_env()
            next_state, _, _, _ = temp_env.step(move)
            state_hash = self._state_hash(next_state)
            
            if state_hash not in self.history[-5:]:
                return move
        
        return move_scores[0][0] if move_scores else None
    
    def _score_move(self, move):
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
            score -= 200
        
        # Heuristic 5: Prefer moving from mixed bottles
        mixed_count = self._count_different_colors(from_idx)
        score += mixed_count * 10
        
        # Heuristic 6: Moving to empty bottle
        if np.sum(to_bottle > 0) == 0:
            if mixed_count <= 2:
                score += 20
            else:
                score -= 10
        
        # Heuristic 7: Consolidating colors
        if self._consolidates_color(from_idx, to_idx):
            score += 15
        
        # Heuristic 8: Amount moved
        amount_moved = self._get_pour_amount(from_idx, to_idx)
        score += amount_moved * 5
        
        return score
    
    def _move_completes_bottle(self, from_idx, to_idx):
        to_bottle = self.env.bottles[to_idx]
        to_count = np.sum(to_bottle > 0)
        if to_count == 0:
            return False
        
        pour_amount = self._get_pour_amount(from_idx, to_idx)
        return to_count + pour_amount == self.env.bottle_height
    
    def _move_to_same_color(self, from_idx, to_idx):
        to_bottle = self.env.bottles[to_idx]
        return np.sum(to_bottle > 0) > 0
    
    def _move_empties_bottle(self, from_idx):
        from_bottle = self.env.bottles[from_idx]
        from_count = np.sum(from_bottle > 0)
        return from_count == self._get_pour_amount(from_idx, from_idx)
    
    def _is_uniform_bottle(self, bottle_idx):
        bottle = self.env.bottles[bottle_idx]
        colors = bottle[bottle > 0]
        if len(colors) == 0:
            return False
        return len(np.unique(colors)) == 1
    
    def _count_different_colors(self, bottle_idx):
        bottle = self.env.bottles[bottle_idx]
        colors = bottle[bottle > 0]
        return len(np.unique(colors))
    
    def _consolidates_color(self, from_idx, to_idx):
        from_bottle = self.env.bottles[from_idx]
        to_bottle = self.env.bottles[to_idx]
        
        if np.sum(to_bottle > 0) == 0:
            return False
        
        from_top = from_bottle[np.where(from_bottle > 0)[0][0]]
        to_top = to_bottle[np.where(to_bottle > 0)[0][0]]
        
        return from_top == to_top
    
    def _get_pour_amount(self, from_idx, to_idx):
        from_bottle = self.env.bottles[from_idx]
        to_bottle = self.env.bottles[to_idx]
        
        source_non_empty = np.where(from_bottle > 0)[0]
        if len(source_non_empty) == 0:
            return 0
        
        source_top_idx = source_non_empty[0]
        source_color = from_bottle[source_top_idx]
        
        pour_amount = 1
        for i in range(source_top_idx + 1, len(from_bottle)):
            if from_bottle[i] == source_color:
                pour_amount += 1
            else:
                break
        
        dest_empty = np.where(to_bottle == 0)[0]
        available_space = len(dest_empty)
        
        return min(pour_amount, available_space)
    
    def _copy_env(self):
        new_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
        new_env.bottles = self.env.bottles.copy()
        return new_env
    
    def _state_hash(self, state):
        return state.tobytes()

# =============================================================================
# 3. DATA GENERATION
# =============================================================================

class HeuristicDataGenerator:
    """Generate training data from heuristic solver"""
    
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.env = WaterSortEnv(num_colors, bottle_height, num_bottles)
        self.solver = HeuristicSolver(self.env)
        self.dataset = []
    
    def generate_data(self, num_games=5000):
        """Generate training data from multiple games"""
        print(f"🎯 Generating {num_games} expert games...")
        
        success_count = 0
        total_moves = 0
        
        pbar = tqdm(range(num_games), desc="Generating data")
        
        for game_idx in pbar:
            state = self.env.reset()
            self.solver.env = self.env
            
            solution, success = self.solver.solve(max_moves=200)
            
            if success:
                success_count += 1
                total_moves += len(solution)
                
                temp_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
                temp_env.bottles = state.copy()
                
                for move_idx, move in enumerate(solution):
                    current_state = temp_env.get_state()
                    
                    policy_target = np.zeros(self.env.num_bottles * self.env.num_bottles)
                    move_index = self._action_to_index(move)
                    policy_target[move_index] = 1.0
                    
                    remaining_moves = len(solution) - move_idx
                    value_target = 1.0 - (remaining_moves / len(solution)) * 0.5
                    
                    self.dataset.append((current_state, policy_target, value_target))
                    
                    temp_env.step(move)
            
            win_rate = success_count / (game_idx + 1)
            avg_moves = total_moves / max(success_count, 1)
            pbar.set_description(f"Win: {win_rate:.1%}, Avg moves: {avg_moves:.1f}")
        
        print(f"\n✓ Generated {len(self.dataset)} training samples")
        print(f"  Success rate: {success_count}/{num_games} ({success_count/num_games:.1%})")
        
        return success_count / num_games
    
    def save_data(self, filename='heuristic_data.pkl'):
        """Save generated data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"✓ Saved {len(self.dataset)} samples to {filename}")
    
    def _action_to_index(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx

# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """Main training function - Generate training data"""
    print("=" * 70)
    print("🎯 PHASE 1: HEURISTIC SOLVER DATA GENERATION")
    print("=" * 70)
    
    generator = HeuristicDataGenerator(num_colors=6, bottle_height=4, num_bottles=8)
    
    # Generate 5,000 games
    win_rate = generator.generate_data(num_games=5000)
    
    # Save data
    generator.save_data('heuristic_data.pkl')
    
    print(f"\n{'='*70}")
    print("✅ PHASE 1 COMPLETED!")
    print(f"   Generated {len(generator.dataset)} training samples")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Saved to: heuristic_data.pkl")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train()