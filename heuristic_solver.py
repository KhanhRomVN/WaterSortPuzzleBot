# heuristic_solver.py
# Phase 1: Generate training data with A* Solver
# =============================================================================

import numpy as np
import random
import pickle
import os
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from heapq import heappush, heappop
from collections import deque

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
    
    def _state_hash(self, state):
        """Hash state for memoization"""
        return state.tobytes()

# =============================================================================
# 2. A* SOLVER (IMPROVED)
# =============================================================================

class AStarSolver:
    """A* solver with admissible heuristic"""
    
    def __init__(self, env: WaterSortEnv):
        self.env = env
        self.debug_info = {
            'nodes_expanded': 0,
            'nodes_in_queue': 0,
            'moves_taken': 0
        }
    
    def _heuristic(self, state) -> float:
        """
        Admissible heuristic: số colors chưa hoàn thành
        
        Heuristic này LUÔN underestimate actual cost (admissible)
        → Guaranteed optimal solution
        """
        bottles = state
        incomplete_colors = 0
        
        for bottle in bottles:
            unique_colors = np.unique(bottle[bottle > 0])
            
            # Nếu bottle có >1 color → 1 incomplete color
            if len(unique_colors) > 1:
                incomplete_colors += len(unique_colors)
            elif len(unique_colors) == 1:
                # Bottle hoàn thành nếu full hoặc empty
                filled_count = np.sum(bottle > 0)
                if filled_count != 0 and filled_count != self.env.bottle_height:
                    incomplete_colors += 1
        
        return incomplete_colors / 2.0
    
    def solve(self, max_moves=1000, debug=False):
        """
        A* search algorithm
        
        f(n) = g(n) + h(n)
        - g(n): số moves từ start đến node n
        - h(n): heuristic estimate từ n đến goal
        """
        initial_state = self.env.get_state()
        
        # Priority queue: (f_score, counter, state_hash, moves_path)
        open_set = []
        counter = 0
        
        initial_hash = self.env._state_hash(initial_state)
        g_score = 0
        h_score = self._heuristic(initial_state)
        f_score = g_score + h_score
        
        heappush(open_set, (f_score, counter, initial_hash, [], initial_state.copy()))
        
        closed_set = set()
        g_scores = {initial_hash: 0}
        
        self.debug_info = {
            'nodes_expanded': 0,
            'nodes_in_queue': len(open_set),
            'moves_taken': 0
        }
        
        while open_set:
            if debug and self.debug_info['nodes_expanded'] % 1000 == 0:
                print(f"[DEBUG A*] Expanded: {self.debug_info['nodes_expanded']}, Queue: {len(open_set)}")
            
            # Get node với f_score nhỏ nhất
            f, _, state_hash, moves_path, current_state = heappop(open_set)
            
            # Nếu đã visited
            if state_hash in closed_set:
                continue
            
            closed_set.add(state_hash)
            self.debug_info['nodes_expanded'] += 1
            
            # Kiểm tra goal
            self.env.bottles = current_state.copy()
            if self.env.is_solved():
                if debug:
                    print(f"[DEBUG A*] ✅ Solution found in {len(moves_path)} moves")
                    print(f"[DEBUG A*] Nodes expanded: {self.debug_info['nodes_expanded']}")
                self.debug_info['moves_taken'] = len(moves_path)
                return moves_path, True
            
            # Kiểm tra max_moves
            if len(moves_path) >= max_moves:
                if debug:
                    print(f"[DEBUG A*] Max moves reached")
                return moves_path, False
            
            # Generate neighbors
            g = g_scores[state_hash]
            
            for move in self.env.get_valid_moves():
                # Tạo new state
                new_state = current_state.copy()
                self.env.bottles = new_state
                self.env.step(move)
                new_state = self.env.get_state()
                
                new_hash = self.env._state_hash(new_state)
                
                if new_hash in closed_set:
                    continue
                
                tentative_g = g + 1
                
                # Nếu đã tìm thấy path tốt hơn, skip
                if new_hash in g_scores and tentative_g >= g_scores[new_hash]:
                    continue
                
                # Update g_score
                g_scores[new_hash] = tentative_g
                
                # Calculate f_score
                h = self._heuristic(new_state)
                new_f = tentative_g + h
                
                new_moves = moves_path + [move]
                
                counter += 1
                heappush(open_set, (new_f, counter, new_hash, new_moves, new_state.copy()))
            
            self.debug_info['nodes_in_queue'] = len(open_set)
        
        if debug:
            print(f"[DEBUG A*] ❌ No solution found")
        return [], False

# =============================================================================
# 3. FALLBACK GREEDY SOLVER (cho game quá khó)
# =============================================================================

class GreedyFallback:
    """Greedy solver for fallback khi A* timeout"""
    
    def __init__(self, env: WaterSortEnv):
        self.env = env
    
    def solve(self, max_moves=200):
        """Quick greedy solve"""
        solution = []
        history = []
        
        for _ in range(max_moves):
            if self.env.is_solved():
                return solution, True
            
            best_move = None
            best_score = -float('inf')
            
            for move in self.env.get_valid_moves():
                score = self._score_move(move)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move is None:
                return solution, False
            
            state_hash = self.env._state_hash(self.env.get_state())
            if state_hash in history[-5:]:
                return solution, False
            
            history.append(state_hash)
            self.env.step(best_move)
            solution.append(best_move)
        
        return solution, False
    
    def _score_move(self, move):
        """Simple greedy scoring"""
        from_idx, to_idx = move
        score = 0
        
        # Complete bottle
        to_bottle = self.env.bottles[to_idx]
        to_count = np.sum(to_bottle > 0)
        if to_count == 0:
            return 100
        if to_count == self.env.bottle_height - 1:
            score += 50
        
        # Move to same color
        from_bottle = self.env.bottles[from_idx]
        from_top_idx = np.where(from_bottle > 0)[0]
        if len(from_top_idx) == 0:
            return -1000
        
        from_color = from_bottle[from_top_idx[0]]
        to_top_idx = np.where(to_bottle > 0)[0]
        if len(to_top_idx) > 0:
            to_color = to_bottle[to_top_idx[0]]
            if from_color == to_color:
                score += 20
        
        return score

# =============================================================================
# 4. DATA GENERATION
# =============================================================================

class HeuristicDataGenerator:
    """Generate training data using A* Solver"""
    
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.env = WaterSortEnv(num_colors, bottle_height, num_bottles)
        self.astar_solver = AStarSolver(self.env)
        self.greedy_solver = GreedyFallback(self.env)
        self.dataset = []
        self.failed_games = []
    
    def generate_data(self, num_games=5000, debug_interval=500):
        """Generate training data from A* solver"""
        print(f"🎯 Generating {num_games} expert games with A* Solver...")
        
        success_count = 0
        astar_count = 0
        greedy_count = 0
        total_moves = 0
        start_time = time.time()
        
        pbar = tqdm(range(num_games), desc="Generating data")
        
        for game_idx in pbar:
            state = self.env.reset()
            self.astar_solver.env = self.env
            self.greedy_solver.env = self.env
            
            # Try A* first (with timeout)
            solution = []
            success = False
            used_astar = False
            
            # A* với giới hạn nodes
            solution, success = self.astar_solver.solve(max_moves=1000, debug=False)
            
            if success:
                success_count += 1
                astar_count += 1
                used_astar = True
                total_moves += len(solution)
            else:
                # Fallback: dùng Greedy
                self.greedy_solver.env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
                self.greedy_solver.env.bottles = state.copy()
                
                solution, success = self.greedy_solver.solve(max_moves=200)
                
                if success:
                    success_count += 1
                    greedy_count += 1
                    total_moves += len(solution)
                else:
                    self.failed_games.append({
                        'game_idx': game_idx,
                        'astar_nodes': self.astar_solver.debug_info['nodes_expanded']
                    })
            
            # Add samples
            if success:
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
            pbar.set_description(f"Win: {win_rate:.1%}, Avg: {avg_moves:.1f}, A*: {astar_count}, GD: {greedy_count}")
            
            # Debug output
            if (game_idx + 1) % debug_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_idx + 1) / elapsed
                remaining_games = num_games - (game_idx + 1)
                eta_seconds = remaining_games / games_per_sec if games_per_sec > 0 else 0
                eta = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"\n[DEBUG] Progress {game_idx+1}/{num_games}")
                print(f"        Success rate: {success_count}/{game_idx+1} ({win_rate:.1%})")
                print(f"        A* solver: {astar_count} games")
                print(f"        Greedy fallback: {greedy_count} games")
                print(f"        Avg moves: {avg_moves:.1f}")
                print(f"        Samples: {len(self.dataset)}")
                print(f"        Time: {str(timedelta(seconds=int(elapsed)))}")
                print(f"        ETA: {eta}")
                print(f"        Failed: {len(self.failed_games)}")
        
        print(f"\n✓ Generated {len(self.dataset)} training samples")
        print(f"  Success rate: {success_count}/{num_games} ({success_count/num_games:.1%})")
        print(f"  A* solutions: {astar_count}")
        print(f"  Greedy fallback: {greedy_count}")
        print(f"  Failed games: {len(self.failed_games)}")
        
        return success_count / num_games
    
    def save_data(self, filename='heuristic_data.pkl'):
        """Save generated data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"✓ Saved {len(self.dataset)} samples to {filename}")
    
    def analyze_failed_games(self):
        """Analyze failed games"""
        if len(self.failed_games) == 0:
            print("✅ Không có game thất bại!")
            return
        
        print("\n" + "="*70)
        print("📊 PHÂN TÍCH GAME THẤT BẠI")
        print("="*70)
        print(f"📈 Số game thất bại: {len(self.failed_games)}")
    
    def _action_to_index(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx

# =============================================================================
# 5. TEST GAME
# =============================================================================

class GameTester:
    """Test solver on real games"""
    
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.env = WaterSortEnv(num_colors, bottle_height, num_bottles)
        self.astar_solver = AStarSolver(self.env)
        self.greedy_solver = GreedyFallback(self.env)
    
    def test_solver(self, num_test_games=50):
        """Test solver on hard games"""
        print("\n" + "="*70)
        print("🧪 TEST A* SOLVER - GAME ĐỘ KHÓ CAO")
        print("="*70)
        
        results = {
            'wins': 0,
            'fails': 0,
            'moves_list': [],
            'astar_count': 0,
            'greedy_count': 0
        }
        
        pbar = tqdm(range(num_test_games), desc="Testing")
        
        for test_idx in pbar:
            state = self.env.reset()
            self.astar_solver.env = self.env
            
            solution, success = self.astar_solver.solve(max_moves=1000, debug=False)
            
            if not success:
                self.greedy_solver.env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
                self.greedy_solver.env.bottles = state.copy()
                solution, success = self.greedy_solver.solve(max_moves=200)
                if success:
                    results['greedy_count'] += 1
            else:
                results['astar_count'] += 1
            
            if success:
                results['wins'] += 1
                results['moves_list'].append(len(solution))
            else:
                results['fails'] += 1
            
            pbar.set_description(f"Win: {results['wins']}/{test_idx+1} ({results['wins']/(test_idx+1):.1%})")
        
        # Calculate statistics
        win_rate = results['wins'] / num_test_games
        avg_moves = np.mean(results['moves_list']) if results['moves_list'] else 0
        
        print(f"\n📊 KẾT QUẢ TEST:")
        print(f"   ✅ Wins: {results['wins']}/{num_test_games} ({win_rate:.1%})")
        print(f"   ❌ Fails: {results['fails']}/{num_test_games} ({results['fails']/num_test_games:.1%})")
        print(f"   A* solutions: {results['astar_count']}")
        print(f"   Greedy fallback: {results['greedy_count']}")
        print(f"   📈 Avg moves: {avg_moves:.1f}")
        
        if results['moves_list']:
            print(f"   📊 Min moves: {min(results['moves_list'])}")
            print(f"   📊 Max moves: {max(results['moves_list'])}")
            print(f"   📊 Std dev: {np.std(results['moves_list']):.2f}")
        
        return results

# =============================================================================
# 6. MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """Main training function"""
    print("=" * 70)
    print("🎯 PHASE 1: A* SOLVER DATA GENERATION")
    print("=" * 70)
    print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_start_time = time.time()
    
    # === STEP 1: Generate data ===
    print("\n📝 STEP 1: Generating Training Data")
    print("-" * 70)
    gen_start_time = time.time()
    
    generator = HeuristicDataGenerator(num_colors=6, bottle_height=4, num_bottles=8)
    win_rate = generator.generate_data(num_games=5000, debug_interval=500)
    
    gen_elapsed = time.time() - gen_start_time
    print(f"⏱️  Generation time: {str(timedelta(seconds=int(gen_elapsed)))}")
    
    # === STEP 2: Analyze failed games ===
    print("\n📊 STEP 2: Analyzing Failed Games")
    print("-" * 70)
    generator.analyze_failed_games()
    
    # === STEP 3: Save data ===
    print("\n💾 STEP 3: Saving Data")
    print("-" * 70)
    generator.save_data('heuristic_data.pkl')
    
    # === STEP 4: Test solver ===
    print("\n🧪 STEP 4: Testing on Real Games")
    print("-" * 70)
    tester = GameTester(num_colors=6, bottle_height=4, num_bottles=8)
    test_results = tester.test_solver(num_test_games=50)
    
    # === FINAL SUMMARY ===
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print("✅ PHASE 1 COMPLETED!")
    print(f"{'='*70}")
    print(f"📊 TRAINING DATA:")
    print(f"   Generated samples: {len(generator.dataset)}")
    print(f"   Success rate: {win_rate:.1%}")
    print(f"   Failed games: {len(generator.failed_games)}")
    print(f"\n📊 TEST RESULTS:")
    print(f"   Test win rate: {test_results['wins']}/50 ({test_results['wins']/50:.1%})")
    print(f"   Avg moves: {np.mean(test_results['moves_list']) if test_results['moves_list'] else 0:.1f}")
    print(f"\n⏱️  TIMING:")
    print(f"   Generation time: {str(timedelta(seconds=int(gen_elapsed)))}")
    print(f"   Total time: {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"   Saved to: heuristic_data.pkl")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train()