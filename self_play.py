# self_play.py
# Phase 3: Generate self-play data with MCTS
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import random
import pickle
from tqdm import tqdm
import time
import os
from collections import deque
from heuristic_solver import WaterSortEnv
from imitation_learning import WaterSortNet, DataProcessor

# =============================================================================
# 1. MONTE CARLO TREE SEARCH (MCTS)
# =============================================================================

class Node:
    """MCTS Node for Water Sort Puzzle"""
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior   # Prior probability from neural network
        
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0
        
    def is_fully_expanded(self):
        """Check if all possible moves have been expanded"""
        return len(self.children) > 0
    
    def is_leaf(self):
        """Check if node has no children"""
        return len(self.children) == 0
    
    def get_ucb_score(self, exploration_weight=1.0):
        """Calculate UCB1 score for node selection"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.mean_value
        exploration = exploration_weight * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def update(self, value):
        """Update node statistics"""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

class MCTS:
    """Monte Carlo Tree Search for Water Sort Puzzle"""
    
    def __init__(self, model, processor, num_simulations=100, exploration_weight=1.0):
        self.model = model
        self.processor = processor
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.env = WaterSortEnv()
        
    def search(self, root_state, temperature=1.0):
        """Perform MCTS search from root state"""
        root = Node(root_state)
        
        for _ in range(self.num_simulations):
            # Selection & Expansion
            node = self._select(root)
            
            # Simulation
            value = self._simulate(node.state)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        return self._get_action_probs(root, temperature)
    
    def _select(self, node):
        """Select node using UCB until leaf node is reached"""
        while node.is_fully_expanded() and not self._is_terminal(node.state):
            # Choose child with highest UCB score
            best_score = -float('inf')
            best_child = None
            
            for action, child in node.children.items():
                score = child.get_ucb_score(self.exploration_weight)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
        
        # Expand if this is a leaf node and not terminal
        if not self._is_terminal(node.state) and not node.is_fully_expanded():
            return self._expand(node)
        
        return node
    
    def _expand(self, node):
        """Expand node by adding children for all valid moves"""
        valid_moves = self._get_valid_moves(node.state)
        
        if not valid_moves:
            return node
        
        # Get prior probabilities from neural network
        state_tensor = self.processor.state_to_tensor(node.state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
        
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Create child nodes for valid moves
        for move in valid_moves:
            action_idx = self._action_to_index(move)
            prior_prob = policy_probs[action_idx]
            
            # Get next state
            next_state = self._get_next_state(node.state, move)
            
            child = Node(
                state=next_state,
                parent=node,
                action=move,
                prior=prior_prob
            )
            
            node.children[action_idx] = child
        
        # Return first child for simulation
        first_action_idx = list(node.children.keys())[0]
        return node.children[first_action_idx]
    
    def _simulate(self, state):
        """Simulate from state using neural network value estimation"""
        state_tensor = self.processor.state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(state_tensor)
        
        return value.item()
    
    def _backpropagate(self, node, value):
        """Backpropagate value up the tree"""
        while node is not None:
            node.update(value)
            node = node.parent
    
    def _get_action_probs(self, root, temperature=1.0):
        """Get action probabilities from root visit counts"""
        action_visits = []
        
        for action_idx, child in root.children.items():
            action_visits.append((action_idx, child.visit_count))
        
        if not action_visits:
            return np.zeros(self.env.num_bottles * self.env.num_bottles), None
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        # Apply temperature
        if temperature == 0:
            # Greedy selection
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)
        
        # Create full probability vector
        full_probs = np.zeros(self.env.num_bottles * self.env.num_bottles)
        for action_idx, prob in zip(actions, probs):
            full_probs[action_idx] = prob
        
        # Choose best action for actual play
        best_action_idx = actions[np.argmax(visits)]
        best_action = self._index_to_action(best_action_idx)
        
        return full_probs, best_action
    
    def _get_valid_moves(self, state):
        """Get valid moves for given state"""
        self.env.bottles = state.copy()
        return self.env.get_valid_moves()
    
    def _get_next_state(self, state, action):
        """Get next state after applying action"""
        self.env.bottles = state.copy()
        next_state, _, _, _ = self.env.step(action)
        return next_state
    
    def _is_terminal(self, state):
        """Check if state is terminal"""
        self.env.bottles = state.copy()
        return self.env.is_solved() or len(self._get_valid_moves(state)) == 0
    
    def _action_to_index(self, action):
        """Convert action to index"""
        from_idx, to_idx = action
        return from_idx * self.env.num_bottles + to_idx
    
    def _index_to_action(self, index):
        """Convert index to action"""
        from_idx = index // self.env.num_bottles
        to_idx = index % self.env.num_bottles
        return (from_idx, to_idx)

# =============================================================================
# 2. SELF-PLAY AGENT
# =============================================================================

class SelfPlayAgent:
    """Agent that plays games using MCTS and collects training data"""
    
    def __init__(self, model, processor, num_simulations=100):
        self.model = model
        self.processor = processor
        self.mcts = MCTS(model, processor, num_simulations)
        self.env = WaterSortEnv()
        self.replay_buffer = deque(maxlen=10000)
        
    def play_game(self, temperature=1.0, max_moves=200):
        """Play one game and collect data"""
        state = self.env.reset()
        game_history = []
        
        for move_idx in range(max_moves):
            # Get MCTS policy
            policy_probs, action = self.mcts.search(state, temperature)
            
            if action is None:
                break
            
            # Store data
            game_history.append((state.copy(), policy_probs))
            
            # Execute move
            next_state, reward, done, _ = self.env.step(action)
            
            if done:
                # Game won
                self._add_game_to_buffer(game_history, 1.0)
                return len(game_history), True
            
            state = next_state
        
        # Game lost or incomplete
        self._add_game_to_buffer(game_history, -1.0)
        return len(game_history), False
    
    def _add_game_to_buffer(self, game_history, final_value):
        """Add game data to replay buffer with value targets"""
        for state, policy in game_history:
            self.replay_buffer.append((state, policy, final_value))
    
    def get_training_data(self, num_samples=None):
        """Get training data from replay buffer"""
        if num_samples is None:
            num_samples = len(self.replay_buffer)
        
        samples = random.sample(self.replay_buffer, min(num_samples, len(self.replay_buffer)))
        return samples

# =============================================================================
# 3. SELF-PLAY DATA GENERATOR
# =============================================================================

class SelfPlayDataGenerator:
    """Generate self-play data using MCTS"""
    
    def __init__(self, model_path=None, num_simulations=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Load model
        self.model = WaterSortNet(num_bottles=8, bottle_height=4, num_colors=6).to(self.device)
        if model_path and os.path.exists(model_path):
            print(f"📁 Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("⚠️  No model found, using random initialization")
        
        self.model.eval()
        self.processor = DataProcessor()
        self.agent = SelfPlayAgent(self.model, self.processor, num_simulations)
        
        self.stats = {
            'games_played': 0,
            'games_won': 0,
            'total_moves': 0,
            'win_rate': 0.0,
            'avg_moves_won': 0.0
        }
    
    def generate_data(self, num_games=100, temperature=1.0):
        """Generate self-play data"""
        print(f"🎯 Generating {num_games} self-play games...")
        
        wins = 0
        total_moves = 0
        moves_won = 0
        
        pbar = tqdm(range(num_games), desc="Self-play")
        
        for game_idx in pbar:
            moves, won = self.agent.play_game(temperature=temperature)
            
            self.stats['games_played'] += 1
            self.stats['total_moves'] += moves
            
            if won:
                wins += 1
                moves_won += moves
                self.stats['games_won'] += 1
            
            # Update stats
            win_rate = wins / (game_idx + 1)
            avg_moves = moves_won / max(wins, 1)
            
            self.stats['win_rate'] = win_rate
            self.stats['avg_moves_won'] = avg_moves
            
            pbar.set_description(f"Win: {win_rate:.1%}, Avg moves: {avg_moves:.1f}")
        
        print(f"\n✅ Generated {len(self.agent.replay_buffer)} self-play samples")
        print(f"   Win rate: {wins}/{num_games} ({win_rate:.1%})")
        print(f"   Avg moves (when won): {avg_moves:.1f}")
        
        return self.agent.get_training_data()
    
    def save_data(self, data, filename='self_play_data.pkl'):
        """Save self-play data to file"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved {len(data)} samples to {filename}")
    
    def evaluate_improvement(self, num_games=50):
        """Evaluate model performance before and after self-play"""
        print("\n🧪 Evaluating model improvement...")
        
        # Test with greedy policy (temperature=0)
        wins_greedy = 0
        moves_greedy = 0
        
        for _ in tqdm(range(num_games), desc="Greedy evaluation"):
            moves, won = self.agent.play_game(temperature=0.0)
            if won:
                wins_greedy += 1
                moves_greedy += moves
        
        win_rate_greedy = wins_greedy / num_games
        avg_moves_greedy = moves_greedy / max(wins_greedy, 1)
        
        print(f"🎯 Greedy policy results:")
        print(f"   Win rate: {wins_greedy}/{num_games} ({win_rate_greedy:.1%})")
        print(f"   Avg moves: {avg_moves_greedy:.1f}")
        
        return win_rate_greedy, avg_moves_greedy

# =============================================================================
# 4. MAIN SELF-PLAY FUNCTION
# =============================================================================

def run_self_play():
    """Main self-play data generation function"""
    print("=" * 70)
    print("🎯 PHASE 3: SELF-PLAY DATA GENERATION")
    print("=" * 70)
    
    # Initialize generator with pre-trained model
    generator = SelfPlayDataGenerator(
        model_path='watersort_imitation.pth',
        num_simulations=100
    )
    
    # Generate self-play data
    self_play_data = generator.generate_data(
        num_games=100,
        temperature=1.0  # Higher temperature for exploration
    )
    
    # Save data
    generator.save_data(self_play_data, 'self_play_data.pkl')
    
    # Evaluate improvement
    win_rate, avg_moves = generator.evaluate_improvement(num_games=50)
    
    print(f"\n{'='*70}")
    print("✅ PHASE 3 COMPLETED!")
    print(f"   Generated {len(self_play_data)} self-play samples")
    print(f"   Win rate (greedy): {win_rate:.1%}")
    print(f"   Avg moves: {avg_moves:.1f}")
    print(f"   Saved to: self_play_data.pkl")
    print(f"{'='*70}\n")
    
    return win_rate, avg_moves

if __name__ == "__main__":
    run_self_play()