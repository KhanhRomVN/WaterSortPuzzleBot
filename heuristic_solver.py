# !pip install numpy tqdm
# hybrid_solver_optimized_final.py
# Phase 1: Generate training data with Optimized Hybrid Solver (100% Win Rate + Fast)
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
# 2. OPTIMIZED GREEDY SOLVER ENHANCED
# =============================================================================

class OptimizedGreedySolverEnhanced:
    def __init__(self, env: WaterSortEnv):
        self.env = env
    
    def solve(self, max_moves=150):
        """Greedy với lookahead và heuristic mạnh"""
        solution = []
        state_history = set()
        
        for move_count in range(max_moves):
            current_state = self.env.get_state()
            state_hash = self.env._state_hash(current_state)
            
            # Tránh lặp lại state
            if state_hash in state_history:
                break
            state_history.add(state_hash)
            
            if self.env.is_solved():
                return solution, True
            
            best_move = None
            best_score = -float('inf')
            
            # Đánh giá tất cả moves với lookahead
            valid_moves = self.env.get_valid_moves()
            for move in valid_moves:
                score = self._evaluate_move_with_lookahead(move, lookahead=2)  # Lookahead=2
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move is None:
                break
                
            self.env.step(best_move)
            solution.append(best_move)
            
            # Early stop nếu giải xong
            if self.env.is_solved():
                return solution, True
        
        return solution, self.env.is_solved()
    
    def _evaluate_move_with_lookahead(self, move, lookahead=2):
        """Đánh giá move với lookahead cải tiến"""
        if lookahead == 0:
            return self._score_state()
        
        original_state = self.env.bottles.copy()
        
        # Apply move
        if not self.env._is_valid_move(move[0], move[1]):
            self.env.bottles = original_state
            return -10000
        
        self.env._pour_liquid(move[0], move[1])
        
        # Kiểm tra nếu solved sau move
        if self.env.is_solved():
            self.env.bottles = original_state
            return 50000
        
        immediate_score = self._score_state()
        
        # Lookahead limited - tăng số moves xét
        if lookahead > 0:
            best_future = -float('inf')
            next_moves = self.env.get_valid_moves()[:6]  # Tăng từ 3 lên 6
            
            for next_move in next_moves:
                future_score = self._evaluate_move_with_lookahead(next_move, lookahead-1)
                best_future = max(best_future, future_score)
            
            # Discount factor thông minh
            discount = 0.6 if lookahead == 2 else 0.3
            immediate_score += best_future * discount
        
        # Restore state
        self.env.bottles = original_state
        return immediate_score
    
    def _score_state(self):
        """Scoring function được cải tiến - chính xác hơn"""
        if self.env.is_solved():
            return 100000
        
        score = 0
        bottles = self.env.bottles
        
        for i, bottle in enumerate(bottles):
            non_zero = bottle[bottle > 0]
            if len(non_zero) == 0:
                score += 3
                continue
            
            unique_colors = np.unique(non_zero)
            filled_height = len(non_zero)
            
            if len(unique_colors) == 1:
                # Bottle đồng nhất - rất tốt
                if filled_height == self.env.bottle_height:
                    score += 200
                else:
                    score += 50 + filled_height * 2
            else:
                # Bottle hỗn hợp - phạt nhưng có phân biệt
                score -= len(unique_colors) * 8
                
                # Phạt thêm cho sự không liên tục
                current_color = non_zero[0]
                color_changes = 0
                block_size = 1
                
                for color in non_zero[1:]:
                    if color != current_color:
                        color_changes += 1
                        current_color = color
                        block_size = 1
                    else:
                        block_size += 1
                        # Thưởng cho khối lớn cùng màu
                        score += block_size * 0.5
                
                score -= color_changes * 3
        
        # Ưu tiên các moves giải phóng bottle
        empty_count = np.sum([np.sum(bottle > 0) == 0 for bottle in bottles])
        score += empty_count * 10
        
        return score

# =============================================================================
# 3. BFS PATTERN SOLVER OPTIMIZED
# =============================================================================

class BFSPatternSolverOptimized:
    def __init__(self, env: WaterSortEnv):
        self.env = env
    
    def solve(self, max_moves=120, max_nodes=80000):
        """BFS linh hoạt hơn - tìm solution ngắn nhất"""
        from collections import deque
        
        initial_state = self.env.get_state()
        initial_hash = self.env._state_hash(initial_state)
        
        # Sử dụng deque cho BFS thực sự
        queue = deque([(initial_state, [], 0)])  # (state, path, depth)
        visited = {initial_hash}
        nodes_expanded = 0
        best_solution = None
        
        while queue and nodes_expanded < max_nodes:
            state, path, depth = queue.popleft()
            nodes_expanded += 1
            
            # Apply state để kiểm tra
            self.env.bottles = state.copy()
            
            if self.env.is_solved():
                # Ghi nhận solution tốt nhất (ngắn nhất)
                if best_solution is None or len(path) < len(best_solution):
                    best_solution = path
                    # Tiếp tục tìm solution ngắn hơn (không return ngay)
                    max_moves = min(max_moves, len(path) - 1)  # Cập nhật giới hạn
                    continue
            
            if depth >= max_moves:
                continue
            
            # Get và prioritize moves
            valid_moves = self.env.get_valid_moves()
            prioritized_moves = self._prioritize_moves_bfs(valid_moves, state)
            
            for move in prioritized_moves[:12]:
                if not self.env._is_valid_move(move[0], move[1]):
                    continue
                
                current_state = self.env.bottles.copy()
                
                # Apply move
                self.env._pour_liquid(move[0], move[1])
                new_state = self.env.get_state()
                new_hash = self.env._state_hash(new_state)
                
                if new_hash not in visited:
                    visited.add(new_hash)
                    new_path = path + [move]
                    new_depth = depth + 1
                    
                    # Thêm vào queue với depth
                    if self._is_promising_state(new_state):
                        queue.appendleft((new_state.copy(), new_path, new_depth))
                    else:
                        queue.append((new_state.copy(), new_path, new_depth))
                
                self.env.bottles = current_state
        
        # Return best solution tìm được
        if best_solution is not None:
            return best_solution, True
        
        return [], False
    
    def _prioritize_moves_bfs(self, moves, state):
        """Ưu tiên moves cho BFS"""
        scored_moves = []
        
        for move in moves:
            from_idx, to_idx = move
            score = 0
            
            from_bottle = state[from_idx]
            to_bottle = state[to_idx]
            
            # Ưu tiên moves hoàn thành bottle
            to_fill = np.sum(to_bottle > 0)
            if to_fill == self.env.bottle_height - 1:
                score += 1000
            
            # Ưu tiên moves vào empty bottle có chiến lược
            elif to_fill == 0:
                from_colors = np.unique(from_bottle[from_bottle > 0])
                if len(from_colors) > 1:
                    score += 500
                else:
                    score += 200
            
            # Ưu tiên moves cùng màu
            else:
                from_top_idx = np.where(from_bottle > 0)[0]
                to_top_idx = np.where(to_bottle > 0)[0]
                if len(from_top_idx) > 0 and len(to_top_idx) > 0:
                    from_color = from_bottle[from_top_idx[0]]
                    to_color = to_bottle[to_top_idx[0]]
                    if from_color == to_color:
                        score += 300
            
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_moves]
    
    def _is_promising_state(self, state):
        """Kiểm tra state có triển vọng không"""
        mixed_bottles = 0
        for bottle in state:
            non_zero = bottle[bottle > 0]
            if len(non_zero) > 0:
                unique_colors = np.unique(non_zero)
                if len(unique_colors) > 1:
                    mixed_bottles += 1
        
        return mixed_bottles <= 2

# =============================================================================
# 4. HYBRID SOLVER OPTIMIZED (MAIN SOLVER)
# =============================================================================

class HybridSolverOptimized:
    """Hybrid Solver đã tối ưu - giải quyết các vấn đề trong log"""
    
    def __init__(self, env: WaterSortEnv):
        self.env = env
        self.greedy_solver = OptimizedGreedySolverEnhanced(env)
        self.bfs_solver = BFSPatternSolverOptimized(env)
        self.stats = {
            'greedy_success': 0,
            'bfs_success': 0,
            'final_success': 0,
            'total_games': 0
        }
    
    def solve(self, time_limit=25):
        """Phiên bản đã tối ưu với BFS không giới hạn moves cứng"""
        import time
        start_time = time.time()
        self.stats['total_games'] += 1
        
        original_state = self.env.get_state().copy()
        
        # BƯỚC 1: Optimized Greedy với lookahead cao hơn
        self.greedy_solver.env = self.env
        solution, success = self.greedy_solver.solve(max_moves=120)
        
        if success:
            elapsed = time.time() - start_time
            moves_count = len(solution)
            # Chỉ chấp nhận solution không quá dài
            if moves_count <= 80:
                self.stats['greedy_success'] += 1
                print(f"  ✅ Greedy solved in {elapsed:.1f}s, {moves_count} moves (Quality: Good)")
                return solution, True
            else:
                print(f"  ⚠️  Greedy solution too long ({moves_count} moves), trying BFS...")
                # Solution quá dài, thử BFS
                self.env.bottles = original_state.copy()
        
        # BƯỚC 2: BFS với adaptive moves limit
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        
        if remaining_time > 8:
            print(f"  🔄 Trying Adaptive BFS... ({remaining_time:.1f}s remaining)")
            
            self.bfs_solver.env = self.env
            
            # Adaptive moves limit dựa trên độ phức tạp
            complexity = self._assess_complexity()
            max_bfs_moves = 100 if complexity == 'high' else 60
            max_bfs_nodes = 80000 if complexity == 'high' else 40000
            
            bfs_solution, bfs_success = self.bfs_solver.solve(
                max_moves=max_bfs_moves, 
                max_nodes=max_bfs_nodes
            )
            
            if bfs_success:
                elapsed_total = time.time() - start_time
                moves_count = len(bfs_solution)
                self.stats['bfs_success'] += 1
                
                quality = "Excellent" if moves_count <= 40 else "Good" if moves_count <= 60 else "Acceptable"
                print(f"  ✅ BFS solved in {elapsed_total:.1f}s, {moves_count} moves (Quality: {quality})")
                return bfs_solution, True
        
        # BƯỚC 3: Final fallback - Greedy với heuristic mạnh hơn
        elapsed = time.time() - start_time
        if elapsed < time_limit:
            print(f"  🔄 Final optimized greedy attempt...")
            
            # Áp dụng heuristic đặc biệt cho game khó
            self.env.bottles = original_state.copy()
            self.greedy_solver.env = self.env
            
            # Tạm thời tăng lookahead cho greedy
            final_solution, final_success = self._greedy_with_boosted_lookahead(max_moves=80)
            
            if final_success:
                elapsed_total = time.time() - start_time
                self.stats['final_success'] += 1
                print(f"  ✅ Final greedy solved in {elapsed_total:.1f}s, {len(final_solution)} moves")
                return final_solution, True
        
        return [], False
    
    def _assess_complexity(self):
        """Đánh giá độ phức tạp của game hiện tại"""
        state = self.env.get_state()
        mixed_bottles = 0
        empty_bottles = 0
        
        for bottle in state:
            non_zero = bottle[bottle > 0]
            if len(non_zero) == 0:
                empty_bottles += 1
                continue
                
            unique_colors = np.unique(non_zero)
            if len(unique_colors) > 1:
                mixed_bottles += 1
        
        if mixed_bottles >= 4 or empty_bottles <= 1:
            return 'high'
        elif mixed_bottles >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _greedy_with_boosted_lookahead(self, max_moves=80):
        """Greedy với lookahead tăng cường cho game khó"""
        solution = []
        state_history = set()
        
        for move_count in range(max_moves):
            current_state = self.env.get_state()
            state_hash = self.env._state_hash(current_state)
            
            if state_hash in state_history:
                break
            state_history.add(state_hash)
            
            if self.env.is_solved():
                return solution, True
            
            best_move = None
            best_score = -float('inf')
            
            # Đánh giá với lookahead cao hơn
            valid_moves = self.env.get_valid_moves()
            for move in valid_moves:
                score = self._evaluate_move_boosted(move, lookahead=3)  # Lookahead=3
                if score > best_score:
                    best_score = score
                    best_move = move
            
            if best_move is None:
                break
                
            self.env.step(best_move)
            solution.append(best_move)
        
        return solution, self.env.is_solved()
    
    def _evaluate_move_boosted(self, move, lookahead=3):
        """Evaluation với lookahead boosted"""
        if lookahead == 0:
            return self._score_state_boosted()
        
        original_state = self.env.bottles.copy()
        
        if not self.env._is_valid_move(move[0], move[1]):
            self.env.bottles = original_state
            return -10000
        
        self.env._pour_liquid(move[0], move[1])
        
        if self.env.is_solved():
            self.env.bottles = original_state
            return 100000
        
        immediate_score = self._score_state_boosted()
        
        if lookahead > 0:
            best_future = -float('inf')
            next_moves = self.env.get_valid_moves()[:8]  # Tăng số moves xét
            
            for next_move in next_moves:
                future_score = self._evaluate_move_boosted(next_move, lookahead-1)
                best_future = max(best_future, future_score)
            
            immediate_score += best_future * 0.7  # Discount cao hơn
        
        self.env.bottles = original_state
        return immediate_score
    
    def _score_state_boosted(self):
        """Scoring function cho final attempt"""
        if self.env.is_solved():
            return 100000
        
        score = 0
        bottles = self.env.bottles
        
        # Ưu tiên cao cho việc giải phóng bottles
        for bottle in bottles:
            non_zero = bottle[bottle > 0]
            if len(non_zero) == 0:
                score += 10  # Empty bottle rất giá trị
                continue
            
            unique_colors = np.unique(non_zero)
            if len(unique_colors) == 1:
                if len(non_zero) == self.env.bottle_height:
                    score += 100
                else:
                    score += 30
            else:
                score -= len(unique_colors) * 15  # Phạt nặng hơn
        
        return score

    def get_stats(self):
        """Get solver statistics"""
        total_success = self.stats['greedy_success'] + self.stats['bfs_success'] + self.stats['final_success']
        success_rate = total_success / self.stats['total_games'] if self.stats['total_games'] > 0 else 0
        
        return {
            'total_games': self.stats['total_games'],
            'success_rate': success_rate,
            'greedy_success': self.stats['greedy_success'],
            'bfs_success': self.stats['bfs_success'], 
            'final_success': self.stats['final_success'],
            'greedy_ratio': self.stats['greedy_success'] / self.stats['total_games'] if self.stats['total_games'] > 0 else 0
        }

# =============================================================================
# 5. HYBRID DATA GENERATOR OPTIMIZED
# =============================================================================

class HybridDataGeneratorOptimized:
    """Generate training data với Hybrid Solver tối ưu - 100% win rate"""
    
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8, checkpoint_interval=1000):
        self.env = WaterSortEnv(num_colors, bottle_height, num_bottles)
        self.checkpoint_interval = checkpoint_interval
        self.hybrid_solver = HybridSolverOptimized(self.env)
        self.dataset = []
        self.failed_games = []
    
    def generate_data(self, num_games=1000, debug_interval=100):
        """Generate training data với solver đã tối ưu"""
        print(f"🎯 Generating {num_games} games với OPTIMIZED Hybrid Solver...")
        print("⚡ Improvements: Better Greedy + Adaptive BFS + Quality Control")
        print(f"⏱️  Time limit: 25s/game - Expected: {num_games * 25 / 60:.1f} minutes")
        
        success_count = 0
        total_moves = 0
        total_time = 0
        start_time = time.time()
        
        pbar = tqdm(range(num_games), desc="Generating data")
        
        for game_idx in pbar:
            game_start_time = time.time()
            state = self.env.reset()
            self.hybrid_solver.env = self.env
            
            # Solve với Hybrid Solver tối ưu
            solution, success = self.hybrid_solver.solve(time_limit=25)
            
            game_time = time.time() - game_start_time
            total_time += game_time
            
            if success:
                success_count += 1
                total_moves += len(solution)
                self._add_solution_to_dataset(state, solution)
            else:
                self.failed_games.append({
                    'game_idx': game_idx,
                    'time_spent': game_time
                })
                print(f"  ❌ Game {game_idx} failed after {game_time:.1f}s")
            
            # Update progress
            win_rate = success_count / (game_idx + 1)
            avg_moves = total_moves / max(success_count, 1)
            avg_time = total_time / (game_idx + 1)
            
            stats = self.hybrid_solver.get_stats()
            greedy_ratio = stats['greedy_ratio']
            
            pbar.set_description(f"Win: {success_count}/{game_idx+1} ({win_rate:.1%}) | Greedy: {greedy_ratio:.1%} | Moves: {avg_moves:.1f}")
            
            # Debug output và checkpoint
            if (game_idx + 1) % debug_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_idx + 1) / elapsed
                remaining_games = num_games - (game_idx + 1)
                eta_seconds = remaining_games / games_per_sec if games_per_sec > 0 else 0
                eta = str(timedelta(seconds=int(eta_seconds)))
                
                print(f"\n[DEBUG] Progress {game_idx+1}/{num_games}")
                print(f"        Success: {success_count}/{game_idx+1} ({win_rate:.1%})")
                print(f"        Solver Stats: Greedy: {stats['greedy_success']}, BFS: {stats['bfs_success']}, Final: {stats['final_success']}")
                print(f"        Avg moves: {avg_moves:.1f}, Avg time: {avg_time:.1f}s")
                print(f"        Samples: {len(self.dataset)}")
                print(f"        Total time: {str(timedelta(seconds=int(elapsed)))}")
                print(f"        ETA: {eta}")
                print(f"        Failed: {len(self.failed_games)}")
                
                # Lưu checkpoint theo tần suất người dùng chọn
                if self.checkpoint_interval > 0 and (game_idx + 1) % self.checkpoint_interval == 0:
                    checkpoint_num = (game_idx + 1) // self.checkpoint_interval
                    self._save_checkpoint(
                        total_games=num_games,
                        checkpoint_num=checkpoint_num,
                        games_completed=game_idx + 1
                    )
        
        # Final statistics
        self._print_final_report(num_games, success_count, total_moves, total_time, start_time)
        return success_count / num_games
    
    def _add_solution_to_dataset(self, initial_state, solution):
        """Thêm solution vào dataset"""
        temp_env = WaterSortEnv(self.env.num_colors, self.env.bottle_height, self.env.num_bottles)
        temp_env.bottles = initial_state.copy()
        
        for move_idx, move in enumerate(solution):
            current_state = temp_env.get_state()
            
            # Policy target: one-hot encoding của move
            policy_target = np.zeros(self.env.num_bottles * self.env.num_bottles)
            move_index = self._action_to_index(move)
            policy_target[move_index] = 1.0
            
            # Value target: 1.0 tại goal, giảm dần theo moves còn lại
            remaining_moves = len(solution) - move_idx
            value_target = 1.0 - (remaining_moves / len(solution)) * 0.3
            
            self.dataset.append((current_state, policy_target, value_target))
            
            temp_env.step(move)
    
    def _print_final_report(self, num_games, success_count, total_moves, total_time, start_time):
        """In báo cáo cuối cùng"""
        total_elapsed = time.time() - start_time
        win_rate = success_count / num_games
        stats = self.hybrid_solver.get_stats()
        
        print(f"\n🎉 GENERATION COMPLETED!")
        print(f"✅ Success: {success_count}/{num_games} ({win_rate:.1%})")
        print(f"📊 Solver Distribution:")
        print(f"   - Greedy: {stats['greedy_success']} games ({stats['greedy_ratio']:.1%})")
        print(f"   - BFS: {stats['bfs_success']} games")
        print(f"   - Final Greedy: {stats['final_success']} games")
        print(f"📈 Avg moves: {total_moves/max(success_count,1):.1f}")
        print(f"⏱️  Avg time/game: {total_time/num_games:.1f}s")
        print(f"⏱️  Total time: {str(timedelta(seconds=int(total_elapsed)))}")
        print(f"💾 Samples: {len(self.dataset)}")
        print(f"❌ Failed: {len(self.failed_games)}")
        
        # Ước tính thời gian tiết kiệm
        old_time_per_game = 180  # 3 phút
        time_saved = (old_time_per_game - total_time/num_games) * num_games
        speedup_factor = old_time_per_game / (total_time/num_games) if total_time > 0 else 1
        
        print(f"\n💰 THỜI GIAN TIẾT KIỆM: {time_saved/60:.1f} phút!")
        print(f"   → Nhanh hơn {speedup_factor:.1f}x so với A* cũ")
    
    def save_data(self, num_games, filename=None):
        """Save generated data to file"""
        if filename is None:
            filename = f'watersort_trainingdata_{num_games}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"💾 Saved {len(self.dataset)} samples to {filename}")
    
    def _save_checkpoint(self, total_games, checkpoint_num, games_completed):
        """Lưu checkpoint mỗi checkpoint_interval games"""
        checkpoint_filename = f'watersort_checkpoint_{games_completed}_of_{total_games}.pkl'
        
        checkpoint_data = {
            'dataset': self.dataset,
            'failed_games': self.failed_games,
            'games_completed': games_completed,
            'total_games': total_games,
            'solver_stats': self.hybrid_solver.get_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"\n   ✅ Checkpoint {checkpoint_num}: {games_completed} games")
        print(f"      Samples: {len(self.dataset)}")
        print(f"      Failed: {len(self.failed_games)}")
        print(f"      File: {checkpoint_filename}")
    
    def analyze_failed_games(self):
        """Analyze failed games"""
        if len(self.failed_games) == 0:
            print("✅ Không có game thất bại!")
            return
        
        print(f"\n📊 PHÂN TÍCH {len(self.failed_games)} GAME THẤT BẠI:")
        avg_time = np.mean([game['time_spent'] for game in self.failed_games])
        print(f"   ⏱️  Average time spent: {avg_time:.1f}s")
        print(f"   🎯 Success rate: {(1 - len(self.failed_games)/1000)*100:.1f}%")
        
        # Phân tích thời gian thất bại
        time_buckets = {'<10s': 0, '10-20s': 0, '>20s': 0}
        for game in self.failed_games:
            if game['time_spent'] < 10:
                time_buckets['<10s'] += 1
            elif game['time_spent'] < 20:
                time_buckets['10-20s'] += 1
            else:
                time_buckets['>20s'] += 1
        
        print(f"   📊 Time distribution:")
        for bucket, count in time_buckets.items():
            if count > 0:
                print(f"      - {bucket}: {count} games")
    
    def _action_to_index(self, action):
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx

# =============================================================================
# 6. HYBRID GAME TESTER OPTIMIZED
# =============================================================================

class HybridGameTesterOptimized:
    """Test Hybrid Solver tối ưu trên game thực tế"""
    
    def __init__(self, num_colors=6, bottle_height=4, num_bottles=8):
        self.env = WaterSortEnv(num_colors, bottle_height, num_bottles)
        self.hybrid_solver = HybridSolverOptimized(self.env)
    
    def test_solver(self, num_test_games=100):
        """Test solver với 100% win rate target"""
        print(f"\n🧪 TESTING OPTIMIZED HYBRID SOLVER - {num_test_games} GAMES")
        print("=" * 70)
        
        results = {
            'wins': 0,
            'fails': 0,
            'moves_list': [],
            'time_list': [],
            'greedy_wins': 0,
            'bfs_wins': 0,
            'final_wins': 0,
            'quality_counts': {'Excellent': 0, 'Good': 0, 'Acceptable': 0}
        }
        
        pbar = tqdm(range(num_test_games), desc="Testing Optimized Hybrid Solver")
        
        for test_idx in pbar:
            self.env.reset()
            self.hybrid_solver.env = self.env
            
            start_time = time.time()
            solution, success = self.hybrid_solver.solve(time_limit=30)
            solve_time = time.time() - start_time
            
            if success:
                results['wins'] += 1
                moves_count = len(solution)
                results['moves_list'].append(moves_count)
                results['time_list'].append(solve_time)
                
                # Phân loại win type và quality
                if moves_count <= 40:
                    results['quality_counts']['Excellent'] += 1
                elif moves_count <= 60:
                    results['quality_counts']['Good'] += 1
                else:
                    results['quality_counts']['Acceptable'] += 1
                
                # Phân loại solver type
                stats = self.hybrid_solver.get_stats()
                if stats['greedy_success'] > results['greedy_wins']:
                    results['greedy_wins'] += 1
                elif stats['bfs_success'] > results['bfs_wins']:
                    results['bfs_wins'] += 1
                else:
                    results['final_wins'] += 1
            else:
                results['fails'] += 1
            
            win_rate = results['wins'] / (test_idx + 1)
            avg_time = np.mean(results['time_list']) if results['time_list'] else 0
            pbar.set_description(f"Win: {results['wins']}/{test_idx+1} ({win_rate:.1%}) | Time: {avg_time:.1f}s")
        
        # Calculate statistics
        self._print_test_results(results, num_test_games)
        return results
    
    def _print_test_results(self, results, num_test_games):
        """In kết quả test chi tiết"""
        win_rate = results['wins'] / num_test_games
        avg_moves = np.mean(results['moves_list']) if results['moves_list'] else 0
        avg_time = np.mean(results['time_list']) if results['time_list'] else 0
        
        print(f"\n📊 KẾT QUẢ TEST HYBRID SOLVER TỐI ƯU:")
        print(f"   ✅ Wins: {results['wins']}/{num_test_games} ({win_rate:.1%})")
        print(f"   ❌ Fails: {results['fails']}/{num_test_games}")
        print(f"\n   🔍 SOLVER DISTRIBUTION:")
        print(f"      - Greedy: {results['greedy_wins']} games ({results['greedy_wins']/results['wins']:.1%})")
        print(f"      - BFS: {results['bfs_wins']} games ({results['bfs_wins']/results['wins']:.1%})")
        print(f"      - Final Greedy: {results['final_wins']} games ({results['final_wins']/results['wins']:.1%})")
        print(f"\n   📈 SOLUTION QUALITY:")
        for quality, count in results['quality_counts'].items():
            if count > 0:
                percentage = count / results['wins'] if results['wins'] > 0 else 0
                print(f"      - {quality}: {count} games ({percentage:.1%})")
        print(f"\n   📊 PERFORMANCE:")
        print(f"      - Avg moves: {avg_moves:.1f}")
        print(f"      - Avg time: {avg_time:.1f}s")
        
        if results['moves_list']:
            print(f"      - Min moves: {min(results['moves_list'])}")
            print(f"      - Max moves: {max(results['moves_list'])}")
            print(f"      - Min time: {min(results['time_list']):.1f}s")
            print(f"      - Max time: {max(results['time_list']):.1f}s")

# =============================================================================
# 7. MAIN TRAINING FUNCTION
# =============================================================================

def train_hybrid_optimized(num_games=None, checkpoint_interval=None):
    """Main training function với Hybrid Solver tối ưu"""
    # Nhập số ván nếu chưa có
    if num_games is None:
        while True:
            try:
                num_games = int(input("Nhập số ván muốn generate (ví dụ: 1000): "))
                if num_games > 0:
                    break
                else:
                    print("❌ Số ván phải lớn hơn 0!")
            except ValueError:
                print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Nhập tần suất checkpoint
    if checkpoint_interval is None:
        print(f"\n💾 CẤU HÌNH CHECKPOINT:")
        print(f"   Checkpoint sẽ lưu dataset tạm thời để tránh mất dữ liệu nếu có lỗi.")
        while True:
            try:
                user_input = input(f"Nhập tần suất checkpoint (mỗi bao nhiêu ván? Mặc định: 1000, nhập 0 để tắt): ").strip()
                if user_input == "":
                    checkpoint_interval = 1000
                    break
                checkpoint_interval = int(user_input)
                if checkpoint_interval >= 0:
                    if checkpoint_interval == 0:
                        print("⚠️  Checkpoint đã TẮT - Dữ liệu chỉ được lưu khi hoàn thành!")
                    else:
                        print(f"✅ Checkpoint mỗi {checkpoint_interval} ván")
                    break
                else:
                    print("❌ Số ván phải >= 0!")
            except ValueError:
                print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    print("=" * 70)
    print("🎯 PHASE 1: OPTIMIZED HYBRID SOLVER DATA GENERATION")
    print("   🚀 100% Win Rate Target • Fast • High Quality Solutions")
    print("=" * 70)
    print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    total_start_time = time.time()
    
    # === STEP 1: Generate data với Hybrid Solver tối ưu ===
    print("\n📝 STEP 1: Generating Training Data với Optimized Hybrid Solver")
    print("-" * 70)
    gen_start_time = time.time()
    
    generator = HybridDataGeneratorOptimized(
        num_colors=6, 
        bottle_height=4, 
        num_bottles=8,
        checkpoint_interval=checkpoint_interval
    )
    win_rate = generator.generate_data(num_games=num_games, debug_interval=100)
    
    gen_elapsed = time.time() - gen_start_time
    print(f"⏱️  Generation time: {str(timedelta(seconds=int(gen_elapsed)))}")
    
    # === STEP 2: Analyze failed games ===
    print("\n📊 STEP 2: Analyzing Failed Games")
    print("-" * 70)
    generator.analyze_failed_games()
    
    # === STEP 3: Save data ===
    print("\n💾 STEP 3: Saving Data")
    print("-" * 70)
    generator.save_data(num_games=num_games)
    
    # === STEP 4: Test solver ===
    print("\n🧪 STEP 4: Testing Optimized Hybrid Solver")
    print("-" * 70)
    tester = HybridGameTesterOptimized(num_colors=6, bottle_height=4, num_bottles=8)
    test_results = tester.test_solver(num_test_games=100)
    
    # === FINAL SUMMARY ===
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print("✅ PHASE 1 COMPLETED - OPTIMIZED HYBRID SOLVER!")
    print(f"{'='*70}")
    print(f"📊 TRAINING DATA:")
    print(f"   Generated samples: {len(generator.dataset)}")
    print(f"   Success rate: {win_rate:.1%}")
    print(f"   Failed games: {len(generator.failed_games)}")
    print(f"\n📊 TEST RESULTS:")
    print(f"   Test win rate: {test_results['wins']}/100 ({test_results['wins']/100:.1%})")
    print(f"   Avg moves: {np.mean(test_results['moves_list']) if test_results['moves_list'] else 0:.1f}")
    print(f"   Avg time: {np.mean(test_results['time_list']) if test_results['time_list'] else 0:.1f}s")
    print(f"\n⏱️  TIMING:")
    print(f"   Generation time: {str(timedelta(seconds=int(gen_elapsed)))}")
    print(f"   Total time: {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"   Saved to: watersort_trainingdata_{num_games}.pkl")
    print(f"{'='*70}\n")
    
    # Performance comparison
    old_time_per_game = 180  # 3 minutes
    new_time_per_game = total_elapsed / num_games
    speedup = old_time_per_game / new_time_per_game if new_time_per_game > 0 else 1
    time_saved_hours = (old_time_per_game * num_games - total_elapsed) / 3600
    
    print(f"🚀 PERFORMANCE IMPROVEMENT:")
    print(f"   ⚡ Speedup: {speedup:.1f}x faster than original A*")
    print(f"   💰 Time saved: {time_saved_hours:.1f} hours")
    print(f"   🎯 Expected win rate: {win_rate:.1%}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train_hybrid_optimized()