# self_play.py - STANDALONE VERSION
# Phase 3: Generate self-play data with MCTS - NO DEPENDENCIES
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
from tqdm import tqdm
import time
import os
import glob
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. FILE PICKER FOR COLAB
# =============================================================================

def select_pth_file_colab():
    """Chọn file .pth trong Colab environment"""
    pth_files = glob.glob('*.pth')
    
    if not pth_files:
        print("❌ Không tìm thấy file .pth nào trong thư mục hiện tại")
        print("📁 Vui lòng upload file .pth trước!")
        return None
    
    print("📁 CÁC FILE PTH CÓ SẴN:")
    print("-" * 70)
    for idx, filename in enumerate(pth_files, 1):
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        print(f"  {idx}. {filename:<40} ({file_size:.1f} MB)")
    
    print("-" * 70)
    while True:
        try:
            choice = int(input("Nhập số thứ tự file (ví dụ: 1): "))
            if 1 <= choice <= len(pth_files):
                selected_file = pth_files[choice - 1]
                print(f"✅ Đã chọn: {selected_file}\n")
                return selected_file
            else:
                print(f"❌ Vui lòng nhập số từ 1 đến {len(pth_files)}")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ")

# =============================================================================
# 1. WATER SORT ENVIRONMENT (STANDALONE)
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
# 2. NEURAL NETWORK ARCHITECTURE (STANDALONE)
# =============================================================================

class WaterSortNet(nn.Module):
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        super(WaterSortNet, self).__init__()
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
        
        # Input: (num_colors, num_bottles, bottle_height)
        self.conv1 = nn.Conv2d(num_colors, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.policy_fc1 = nn.Linear(64 * num_bottles * bottle_height, 512)
        self.policy_fc2 = nn.Linear(512, num_bottles * num_bottles)
        self.policy_bn = nn.BatchNorm1d(512)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.value_fc1 = nn.Linear(64 * num_bottles * bottle_height, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)
        self.value_bn1 = nn.BatchNorm1d(512)
        self.value_bn2 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x shape: (batch_size, num_colors, num_bottles, bottle_height)
        batch_size = x.size(0)
        
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        # Policy head
        policy = self.relu(self.policy_conv(x))
        policy = policy.view(batch_size, -1)
        policy = self.dropout(self.relu(self.policy_bn(self.policy_fc1(policy))))
        policy = self.policy_fc2(policy)
        
        # Value head
        value = self.relu(self.value_conv(x))
        value = value.view(batch_size, -1)
        value = self.dropout(self.relu(self.value_bn1(self.value_fc1(value))))
        value = self.dropout(self.relu(self.value_bn2(self.value_fc2(value))))
        value = torch.tanh(self.value_fc3(value))
        
        return policy, value

# =============================================================================
# 3. DATA PROCESSOR (STANDALONE)
# =============================================================================

class DataProcessor:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
    
    def state_to_tensor(self, state):
        """Chuyển state thành one-hot encoded tensor"""
        one_hot = np.zeros((self.num_colors, self.num_bottles, self.bottle_height), dtype=np.float32)
        
        for bottle_idx in range(self.num_bottles):
            for height_idx in range(self.bottle_height):
                color = int(state[bottle_idx, height_idx])
                if color > 0:
                    one_hot[color - 1, bottle_idx, height_idx] = 1.0
        
        return torch.from_numpy(one_hot)

# =============================================================================
# 4. OPTIMIZED MCTS NODE
# =============================================================================

class MCTSNode:
    """MCTS Node với memory footprint thấp"""
    __slots__ = ['state_hash', 'parent', 'action', 'prior', 'children', 'visit_count', 
                 'total_value', 'mean_value']
    
    def __init__(self, state_hash, parent=None, action=None, prior=0.0):
        self.state_hash = state_hash
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
    
    def get_ucb_score(self, parent_visit_count, exploration_weight=1.414):
        """UCB calculation"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.mean_value
        exploration = exploration_weight * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def update(self, value):
        """Update node với value mới"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

# =============================================================================
# 5. OPTIMIZED MCTS
# =============================================================================

class OptimizedMCTS:
    """MCTS tối ưu với batch evaluation"""
    
    def __init__(self, model, processor, num_simulations=100, device='cuda'):
        self.model = model
        self.processor = processor
        self.num_simulations = num_simulations
        self.device = device
        
        self.env = WaterSortEnv()
        self.nodes_cache = {}
    
    def _hash_state(self, state):
        """Hash state"""
        return state.tobytes()
    
    def search(self, root_state, temperature=1.0):
        """MCTS search"""
        root_hash = self._hash_state(root_state)
        
        if root_hash in self.nodes_cache:
            root = self.nodes_cache[root_hash]
        else:
            root = MCTSNode(root_hash)
            self.nodes_cache[root_hash] = root
        
        # Run simulations
        for _ in range(self.num_simulations):
            node, state = self._select_and_expand(root, root_state)
            value = self._evaluate(state)
            self._backpropagate(node, value)
        
        return self._get_action_probs(root, root_state, temperature)
    
    def _select_and_expand(self, node, root_state):
        """Select và expand node"""
        current_state = root_state.copy()
        path = [node]
        
        while True:
            self.env.bottles = current_state.copy()
            
            # Terminal check
            if self.env.is_solved() or len(self.env.get_valid_moves()) == 0:
                return node, current_state
            
            # Leaf node - expand
            if len(node.children) == 0:
                self._expand_node(node, current_state)
                if len(node.children) == 0:
                    return node, current_state
            
            # Select best child
            best_score = -float('inf')
            best_child = None
            best_action = None
            
            for action_idx, child in node.children.items():
                score = child.get_ucb_score(node.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_action = self._index_to_action(action_idx)
            
            if best_child is None:
                return node, current_state
            
            # Apply action
            next_state = self._get_next_state(current_state, best_action)
            node = best_child
            current_state = next_state
            path.append(node)
        
        return node, current_state
    
    def _expand_node(self, node, state):
        """Expand node"""
        valid_moves = self._get_valid_moves(state)
        
        if not valid_moves:
            return
        
        # Get policy priors
        state_tensor = self.processor.state_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
        
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Create children
        for move in valid_moves:
            action_idx = self._action_to_index(move)
            prior = policy_probs[action_idx]
            
            next_state = self._get_next_state(state, move)
            next_hash = self._hash_state(next_state)
            
            if next_hash not in self.nodes_cache:
                child_node = MCTSNode(next_hash, parent=node, action=move, prior=prior)
                self.nodes_cache[next_hash] = child_node
            else:
                child_node = self.nodes_cache[next_hash]
            
            node.children[action_idx] = child_node
    
    def _evaluate(self, state):
        """Evaluate state bằng neural network"""
        state_tensor = self.processor.state_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(state_tensor)
        
        return value.item()
    
    def _backpropagate(self, node, value):
        """Backpropagation"""
        while node is not None:
            normalized_value = np.tanh(value)
            node.update(normalized_value)
            node = node.parent
    
    def _get_action_probs(self, root, root_state, temperature=1.0):
        """Get action probabilities với temperature"""
        valid_moves = self._get_valid_moves(root_state)
        
        if not valid_moves:
            return np.zeros(64), None
        
        action_visits = []
        for move in valid_moves:
            action_idx = self._action_to_index(move)
            if action_idx in root.children:
                child = root.children[action_idx]
                action_visits.append((action_idx, move, child.visit_count))
        
        if not action_visits:
            return np.zeros(64), None
        
        actions, moves, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        # Apply temperature
        if temperature == 0:
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)
        
        # Full probability vector
        full_probs = np.zeros(64)
        for action_idx, prob in zip(actions, probs):
            full_probs[action_idx] = prob
        
        # Best action
        best_idx = actions[np.argmax(visits)]
        best_action = moves[list(actions).index(best_idx)]
        
        return full_probs, best_action
    
    def _get_valid_moves(self, state):
        """Get valid moves"""
        self.env.bottles = state.copy()
        return self.env.get_valid_moves()
    
    def _get_next_state(self, state, action):
        """Get next state"""
        self.env.bottles = state.copy()
        next_state, _, _, _ = self.env.step(action)
        return next_state
    
    def _action_to_index(self, action):
        """Convert action to index"""
        from_idx, to_idx = action
        return from_idx * 8 + to_idx
    
    def _index_to_action(self, index):
        """Convert index to action"""
        return (index // 8, index % 8)
    
    def clear_cache(self):
        """Clear cache"""
        self.nodes_cache.clear()

# =============================================================================
# 6. SELF-PLAY AGENT
# =============================================================================

class SelfPlayAgent:
    """Self-play agent với MCTS"""
    
    def __init__(self, model, processor, num_simulations=100, device='cuda'):
        self.model = model
        self.processor = processor
        self.mcts = OptimizedMCTS(model, processor, num_simulations, device)
        self.env = WaterSortEnv()
        
        self.replay_buffer = deque(maxlen=20000)
        self.win_buffer = deque(maxlen=5000)
        
    def play_game(self, temperature=1.0, max_moves=200):
        """Play một game"""
        state = self.env.reset()
        game_history = []
        
        for move_idx in range(max_moves):
            # Get MCTS policy
            policy_probs, action = self.mcts.search(state, temperature)
            
            if action is None:
                break
            
            # Store trajectory
            game_history.append((state.copy(), policy_probs.copy()))
            
            # Execute move
            next_state, reward, done, _ = self.env.step(action)
            
            if done:
                self._add_game_to_buffers(game_history, 1.0, is_win=True)
                return len(game_history), True
            
            state = next_state
        
        self._add_game_to_buffers(game_history, -0.5, is_win=False)
        return len(game_history), False
    
    def _add_game_to_buffers(self, game_history, final_value, is_win=False):
        """Add game vào buffers"""
        game_len = len(game_history)
        
        for idx, (state, policy) in enumerate(game_history):
            discount = 0.99 ** (game_len - idx)
            value = final_value * discount
            
            sample = (state, policy, value)
            self.replay_buffer.append(sample)
            
            if is_win:
                self.win_buffer.append(sample)
    
    def get_training_data(self):
        """Get training data"""
        return list(self.replay_buffer)

# =============================================================================
# 7. SELF-PLAY DATA GENERATOR
# =============================================================================

class SelfPlayDataGenerator:
    """Generate self-play data"""
    
    def __init__(self, model_path, num_simulations=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        self.model = WaterSortNet(num_bottles=8, bottle_height=4, num_colors=6).to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            print(f"📁 Loading model từ {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded successfully\n")
        else:
            print("⚠️  No model found, using random initialization\n")
        
        self.processor = DataProcessor()
        self.agent = SelfPlayAgent(self.model, self.processor, num_simulations, self.device)
    
    def generate_data(self, num_games=100, temperature=1.0, adaptive_temp=False):
        """Generate self-play data"""
        print(f"🎯 Generating {num_games} self-play games...")
        print(f"   Temperature: {temperature} | Adaptive: {adaptive_temp}")
        print("-" * 70)
        
        wins = 0
        total_moves = 0
        moves_won = 0
        game_times = []
        
        pbar = tqdm(range(num_games), desc="Self-play generation")
        
        for game_idx in pbar:
            game_start = time.time()
            
            # Adaptive temperature
            if adaptive_temp:
                current_temp = temperature * (1.0 - game_idx / num_games) + 0.1
            else:
                current_temp = temperature
            
            moves, won = self.agent.play_game(temperature=current_temp)
            
            game_time = time.time() - game_start
            game_times.append(game_time)
            
            total_moves += moves
            
            if won:
                wins += 1
                moves_won += moves
            
            # Update progress
            win_rate = wins / (game_idx + 1)
            avg_moves = moves_won / max(wins, 1)
            avg_time = np.mean(game_times)
            
            pbar.set_description(f"Win: {win_rate:.1%} | Moves: {avg_moves:.1f} | Time: {avg_time:.1f}s")
        
        # Final statistics
        total_samples = len(self.agent.replay_buffer)
        total_time = sum(game_times)
        
        print("-" * 70)
        print(f"✅ Generated {total_samples} self-play samples")
        print(f"   Win rate: {wins}/{num_games} ({win_rate:.1%})")
        print(f"   Avg moves (when won): {avg_moves:.1f}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Avg time/game: {avg_time:.2f}s")
        
        return self.agent.get_training_data()
    
    def save_data(self, data, filename='self_play_data.pkl'):
        """Save data"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        file_size = os.path.getsize(filename) / (1024**2)
        print(f"💾 Saved {len(data)} samples to {filename} ({file_size:.1f} MB)")

# =============================================================================
# 8. MAIN FUNCTION
# =============================================================================

def run_self_play():
    """Main self-play generation"""
    print("=" * 70)
    print("🎯 PHASE 3: SELF-PLAY DATA GENERATION (STANDALONE)")
    print("=" * 70 + "\n")
    
    # Select model file
    model_path = select_pth_file_colab()
    if model_path is None:
        return
    
    # Configuration
    print("⚙️  CẤU HÌNH SELF-PLAY:")
    print("-" * 70)
    
    # Số game
    while True:
        try:
            num_games = int(input("Nhập số game muốn generate (mặc định: 100): ") or "100")
            if num_games > 0:
                break
            else:
                print("❌ Số game phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Số simulations
    while True:
        try:
            num_simulations = int(input("Nhập số MCTS simulations (mặc định: 100): ") or "100")
            if num_simulations > 0:
                break
            else:
                print("❌ Số simulations phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Temperature
    while True:
        try:
            temperature = float(input("Nhập temperature (mặc định: 1.0): ") or "1.0")
            if temperature >= 0:
                break
            else:
                print("❌ Temperature phải >= 0!")
        except ValueError:
            print("❌ Vui lòng nhập số thực hợp lệ!")
    
    # Adaptive temperature
    adaptive_temp_input = input("Sử dụng adaptive temperature? (y/n, mặc định: n): ").lower()
    adaptive_temp = adaptive_temp_input == 'y'
    
    print("-" * 70)
    print(f"✅ Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Số games: {num_games}")
    print(f"   MCTS simulations: {num_simulations}")
    print(f"   Temperature: {temperature}")
    print(f"   Adaptive temp: {adaptive_temp}")
    print("=" * 70 + "\n")
    
    # Initialize generator
    generator = SelfPlayDataGenerator(
        model_path=model_path,
        num_simulations=num_simulations
    )
    
    # Generate data
    self_play_data = generator.generate_data(
        num_games=num_games,
        temperature=temperature,
        adaptive_temp=adaptive_temp
    )
    
    # Save data
    output_filename = f'self_play_data_{num_games}games.pkl'
    generator.save_data(self_play_data, output_filename)
    
    print("\n" + "=" * 70)
    print("✅ PHASE 3 COMPLETED!")
    print("=" * 70)
    print(f"📊 Generated {len(self_play_data)} samples")
    print(f"💾 Saved to: {output_filename}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_self_play()