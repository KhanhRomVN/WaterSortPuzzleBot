# reinforcement_learning.py
# Phase 4: Train model with self-play data (AlphaZero style)
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pickle
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from heuristic_solver import WaterSortEnv
from imitation_learning import WaterSortNet, DataProcessor

# =============================================================================
# 1. REINFORCEMENT LEARNING TRAINER
# =============================================================================

class RLTrainer:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Initialize model
        self.model = WaterSortNet(num_bottles, bottle_height, num_colors).to(self.device)
        self.processor = DataProcessor(num_bottles, bottle_height, num_colors)
        
        # Load pre-trained model from imitation learning
        self._load_pretrained_model('watersort_imitation.pth')
        
        # Loss functions
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')
        self.value_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            patience=10, factor=0.5, verbose=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.win_rates = []
        self.best_win_rate = 0.0
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=20000)
    
    def _load_pretrained_model(self, model_path):
        """Load pre-trained model from imitation learning"""
        if os.path.exists(model_path):
            print(f"📁 Loading pre-trained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print("✅ Pre-trained model loaded successfully")
        else:
            print("⚠️  No pre-trained model found, starting from scratch")
    
    def load_self_play_data(self, filename='self_play_data.pkl'):
        """Load and process self-play data"""
        print(f"📁 Loading self-play data from {filename}...")
        with open(filename, 'rb') as f:
            self_play_data = pickle.load(f)
        
        print(f"📊 Loaded {len(self_play_data)} self-play samples")
        
        # Add to replay buffer
        for sample in self_play_data:
            self.replay_buffer.append(sample)
        
        return len(self_play_data)
    
    def prepare_training_data(self, batch_size=32):
        """Prepare training data from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            print("⚠️  Not enough samples in replay buffer")
            return None, None, None
        
        # Sample from replay buffer
        samples = random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size * 10))
        
        states = []
        policy_targets = []
        value_targets = []
        
        for state, policy_target, value_target in samples:
            state_tensor = self.processor.state_to_tensor(state)
            policy_tensor = torch.FloatTensor(policy_target)
            value_tensor = torch.FloatTensor([value_target])
            
            states.append(state_tensor)
            policy_targets.append(policy_tensor)
            value_targets.append(value_tensor)
        
        # Create dataset
        dataset = TensorDataset(
            torch.stack(states), 
            torch.stack(policy_targets), 
            torch.stack(value_targets)
        )
        
        # Split train/validation (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"🎯 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for states, policy_targets, value_targets in tqdm(train_loader, desc="RL Training"):
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Convert policy predictions to log probabilities
            policy_pred_log = torch.log_softmax(policy_pred, dim=1)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_pred_log, policy_targets)
            value_loss = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
            
            # Combined loss (AlphaZero style)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return (total_loss / num_batches, 
                total_policy_loss / num_batches, 
                total_value_loss / num_batches)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for states, policy_targets, value_targets in tqdm(val_loader, desc="RL Validation"):
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Convert policy predictions to log probabilities
                policy_pred_log = torch.log_softmax(policy_pred, dim=1)
                
                # Calculate losses
                policy_loss = self.policy_criterion(policy_pred_log, policy_targets)
                value_loss = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
                
                # Combined loss
                loss = policy_loss + value_loss
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
        
        return (total_loss / num_batches, 
                total_policy_loss / num_batches, 
                total_value_loss / num_batches)
    
    def evaluate_model(self, num_games=100, use_mcts=False):
        """Evaluate model performance"""
        print("🧪 Evaluating model...")
        
        self.model.eval()
        env = WaterSortEnv()
        wins = 0
        total_moves = 0
        
        for game_idx in tqdm(range(num_games), desc="Evaluating"):
            state = env.reset()
            moves = 0
            max_moves = 200
            
            for move_idx in range(max_moves):
                if env.is_solved():
                    wins += 1
                    total_moves += moves
                    break
                
                # Get valid moves
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                
                if use_mcts:
                    # Use MCTS for move selection (better but slower)
                    from self_play import MCTS
                    mcts = MCTS(self.model, self.processor, num_simulations=50)
                    policy_probs, best_move = mcts.search(state, temperature=0.0)
                else:
                    # Use direct policy for move selection (faster)
                    state_tensor = self.processor.state_to_tensor(state).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        policy, value = self.model(state_tensor)
                    
                    policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                    
                    # Select best valid move
                    best_move = None
                    best_score = -float('inf')
                    
                    for move in valid_moves:
                        from_idx, to_idx = move
                        move_index = from_idx * env.num_bottles + to_idx
                        score = policy_probs[move_index]
                        
                        if score > best_score:
                            best_score = score
                            best_move = move
                
                if best_move is None:
                    break
                
                # Execute move
                state, reward, done, _ = env.step(best_move)
                moves += 1
        
        win_rate = wins / num_games
        avg_moves = total_moves / max(wins, 1)
        
        print(f"🎯 Evaluation Results:")
        print(f"   Win rate: {wins}/{num_games} ({win_rate:.1%})")
        print(f"   Avg moves (when win): {avg_moves:.1f}")
        
        return win_rate, avg_moves
    
    def train(self, num_iterations=10, epochs_per_iter=10, batch_size=32, 
              eval_every=2, save_path='watersort_rl_model.pth'):
        """Main reinforcement learning training loop"""
        print("🎯 Starting Reinforcement Learning Training...")
        print(f"   Iterations: {num_iterations}")
        print(f"   Epochs per iteration: {epochs_per_iter}")
        print(f"   Batch size: {batch_size}")
        print(f"   Save path: {save_path}")
        print("-" * 70)
        
        # Load initial self-play data
        self.load_self_play_data('self_play_data.pkl')
        
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")
            
            # Prepare training data
            train_dataset, val_dataset = self.prepare_training_data(batch_size)
            if train_dataset is None:
                print("❌ Not enough training data, skipping iteration")
                continue
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train for multiple epochs
            for epoch in range(epochs_per_iter):
                start_time = time.time()
                
                # Train
                train_loss, train_policy_loss, train_value_loss = self.train_epoch(train_loader)
                
                # Validate
                val_loss, val_policy_loss, val_value_loss = self.validate(val_loader)
                
                epoch_time = time.time() - start_time
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.policy_losses.append(train_policy_loss)
                self.value_losses.append(train_value_loss)
                
                # Print progress
                print(f"  Epoch {epoch+1:2d}/{epochs_per_iter} | "
                      f"Time: {epoch_time:5.1f}s | "
                      f"Train: {train_loss:7.4f} (P: {train_policy_loss:6.4f}, V: {train_value_loss:6.4f}) | "
                      f"Val: {val_loss:7.4f} (P: {val_policy_loss:6.4f}, V: {val_value_loss:6.4f})")
            
            # Step learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Evaluate model periodically
            if (iteration + 1) % eval_every == 0:
                print("\n📊 Running evaluation...")
                win_rate, avg_moves = self.evaluate_model(num_games=50, use_mcts=False)
                self.win_rates.append(win_rate)
                
                # Save best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self._save_model(save_path, iteration, win_rate)
                    print(f"💾 Saved best model (win_rate: {win_rate:.1%})")
                
                # Generate new self-play data with current model
                if win_rate < 0.95:  # Only generate new data if not perfect
                    print("\n🔄 Generating new self-play data...")
                    self._generate_self_play_data(num_games=50)
        
        print("-" * 70)
        print("✅ Reinforcement Learning completed!")
        
        # Final evaluation
        print("\n🎯 FINAL EVALUATION")
        final_win_rate, final_avg_moves = self.evaluate_model(num_games=100, use_mcts=False)
        
        # Save final model
        self._save_model(save_path, num_iterations, final_win_rate, final=True)
        
        return final_win_rate, final_avg_moves
    
    def _generate_self_play_data(self, num_games=50):
        """Generate new self-play data with current model"""
        from self_play import SelfPlayDataGenerator
        
        # Save current model temporarily
        temp_path = 'temp_model.pth'
        self._save_model(temp_path, 0, 0.0)
        
        # Generate self-play data
        generator = SelfPlayDataGenerator(temp_path, num_simulations=50)
        new_data = generator.generate_data(num_games=num_games, temperature=1.0)
        
        # Add to replay buffer
        for sample in new_data:
            self.replay_buffer.append(sample)
        
        print(f"📊 Added {len(new_data)} new samples to replay buffer")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def _save_model(self, save_path, iteration, win_rate, final=False):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': iteration,
            'win_rate': win_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'win_rates': self.win_rates
        }, save_path)
        
        if final:
            print(f"💾 Saved final model to {save_path}")
        else:
            print(f"💾 Saved model checkpoint (iter {iteration}, win_rate: {win_rate:.1%})")
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.train_losses:
            print("No training data to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot policy and value losses
        ax2.plot(epochs, self.policy_losses, 'g-', label='Policy Loss')
        ax2.plot(epochs, self.value_losses, 'm-', label='Value Loss')
        ax2.set_title('Policy and Value Losses')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot win rates
        if self.win_rates:
            eval_points = range(1, len(self.win_rates) + 1)
            ax3.plot(eval_points, self.win_rates, 'c-', marker='o', label='Win Rate')
            ax3.set_title('Win Rate Progress')
            ax3.set_xlabel('Evaluation Point')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True)
        
        # Plot learning rate
        ax4.axis('off')
        ax4.text(0.1, 0.5, f"Best Win Rate: {self.best_win_rate:.1%}\n"
                          f"Final LR: {self.optimizer.param_groups[0]['lr']:.2e}\n"
                          f"Total Epochs: {len(self.train_losses)}", 
                fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

# =============================================================================
# 2. MAIN REINFORCEMENT LEARNING FUNCTION
# =============================================================================

def train_reinforcement_learning():
    """Main reinforcement learning training function"""
    print("=" * 70)
    print("🎯 PHASE 4: REINFORCEMENT LEARNING TRAINING")
    print("=" * 70)
    
    # Initialize trainer
    trainer = RLTrainer(num_bottles=8, bottle_height=4, num_colors=6)
    
    # Train model with reinforcement learning
    final_win_rate, final_avg_moves = trainer.train(
        num_iterations=20,
        epochs_per_iter=5,
        batch_size=32,
        eval_every=2,
        save_path='watersort_rl_model.pth'
    )
    
    # Plot training progress
    trainer.plot_training_progress()
    
    print(f"\n{'='*70}")
    print("✅ PHASE 4 COMPLETED!")
    print(f"   Final model: watersort_rl_model.pth")
    print(f"   Final win rate: {final_win_rate:.1%}")
    print(f"   Final avg moves: {final_avg_moves:.1f}")
    print(f"{'='*70}\n")
    
    return final_win_rate, final_avg_moves

# =============================================================================
# 3. MODEL TESTING FUNCTION
# =============================================================================

def test_model(model_path='watersort_rl_model.pth', num_games=100, use_mcts=True):
    """Test trained model"""
    print("=" * 70)
    print("🧪 MODEL TESTING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaterSortNet(num_bottles=8, bottle_height=4, num_colors=6).to(device)
    processor = DataProcessor()
    
    # Load trained model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {model_path}")
        if 'win_rate' in checkpoint:
            print(f"   Model win rate during training: {checkpoint['win_rate']:.1%}")
    else:
        print(f"❌ Model file {model_path} not found")
        return
    
    model.eval()
    
    # Test with and without MCTS
    print(f"\n🔍 Testing with {'MCTS' if use_mcts else 'direct policy'}...")
    
    env = WaterSortEnv()
    wins = 0
    total_moves = 0
    move_counts = []
    
    if use_mcts:
        from self_play import MCTS
        mcts = MCTS(model, processor, num_simulations=100)
    
    for game_idx in tqdm(range(num_games), desc="Testing"):
        state = env.reset()
        moves = 0
        max_moves = 200
        
        for move_idx in range(max_moves):
            if env.is_solved():
                wins += 1
                total_moves += moves
                move_counts.append(moves)
                break
            
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            
            if use_mcts:
                # Use MCTS
                policy_probs, best_move = mcts.search(state, temperature=0.0)
            else:
                # Use direct policy
                state_tensor = processor.state_to_tensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy, value = model(state_tensor)
                policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                
                best_move = None
                best_score = -float('inf')
                for move in valid_moves:
                    from_idx, to_idx = move
                    move_index = from_idx * env.num_bottles + to_idx
                    score = policy_probs[move_index]
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            if best_move is None:
                break
            
            state, reward, done, _ = env.step(best_move)
            moves += 1
    
    win_rate = wins / num_games
    avg_moves = total_moves / max(wins, 1)
    std_moves = np.std(move_counts) if move_counts else 0
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Win rate: {wins}/{num_games} ({win_rate:.1%})")
    print(f"   Avg moves (when win): {avg_moves:.1f} ± {std_moves:.1f}")
    print(f"   Method: {'MCTS (100 sims)' if use_mcts else 'Direct Policy'}")
    
    return win_rate, avg_moves

if __name__ == "__main__":
    # Run reinforcement learning training
    final_win_rate, final_avg_moves = train_reinforcement_learning()
    
    # Test the final model
    print("\n" + "="*70)
    print("FINAL MODEL TESTING")
    print("="*70)
    
    # Test with direct policy
    test_model('watersort_rl_model.pth', num_games=100, use_mcts=False)
    
    # Test with MCTS (better but slower)
    test_model('watersort_rl_model.pth', num_games=50, use_mcts=True)