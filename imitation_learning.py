# imitation_learning.py
# Phase 2: Train model with Imitation Learning
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
from heuristic_solver import WaterSortEnv

# =============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# =============================================================================

class WaterSortNet(nn.Module):
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        super(WaterSortNet, self).__init__()
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
        
        # Input shape: (num_bottles, bottle_height, num_colors) one-hot encoded
        self.conv1 = nn.Conv2d(num_colors, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Bottle-level features
        self.bottle_conv = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.policy_fc1 = nn.Linear(32 * num_bottles * bottle_height, 512)
        self.policy_fc2 = nn.Linear(512, num_bottles * num_bottles)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.value_fc1 = nn.Linear(32 * num_bottles * bottle_height, 512)
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
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
        policy = self.dropout(self.relu(self.policy_fc1(policy)))
        policy = self.policy_fc2(policy)
        
        # Value head
        value = self.relu(self.value_conv(x))
        value = value.view(batch_size, -1)
        value = self.dropout(self.relu(self.value_fc1(value)))
        value = self.dropout(self.relu(self.value_fc2(value)))
        value = torch.tanh(self.value_fc3(value))
        
        return policy, value

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================

class DataProcessor:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
    
    def state_to_tensor(self, state):
        """Convert state to one-hot encoded tensor"""
        # state shape: (num_bottles, bottle_height)
        one_hot = np.zeros((self.num_colors, self.num_bottles, self.bottle_height))
        
        for bottle_idx in range(self.num_bottles):
            for height_idx in range(self.bottle_height):
                color = state[bottle_idx, height_idx]
                if color > 0:
                    # Colors are 1-indexed, convert to 0-indexed
                    one_hot[color-1, bottle_idx, height_idx] = 1
        
        return torch.FloatTensor(one_hot)
    
    def process_dataset(self, dataset):
        """Process entire dataset"""
        states = []
        policy_targets = []
        value_targets = []
        
        print("🔄 Processing dataset...")
        for state, policy_target, value_target in tqdm(dataset):
            state_tensor = self.state_to_tensor(state)
            policy_tensor = torch.FloatTensor(policy_target)
            value_tensor = torch.FloatTensor([value_target])
            
            states.append(state_tensor)
            policy_targets.append(policy_tensor)
            value_targets.append(value_tensor)
        
        return (torch.stack(states), 
                torch.stack(policy_targets), 
                torch.stack(value_targets))

# =============================================================================
# 3. IMITATION LEARNING TRAINER
# =============================================================================

class ImitationTrainer:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        self.model = WaterSortNet(num_bottles, bottle_height, num_colors).to(self.device)
        self.processor = DataProcessor(num_bottles, bottle_height, num_colors)
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.policy_losses = []
        self.value_losses = []
    
    def load_data(self, filename='heuristic_data.pkl'):
        """Load and process training data"""
        print(f"📁 Loading data from {filename}...")
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"📊 Loaded {len(dataset)} samples")
        
        # Process data
        states, policy_targets, value_targets = self.processor.process_dataset(dataset)
        
        # Create dataset
        full_dataset = TensorDataset(states, policy_targets, value_targets)
        
        # Split train/validation (90/10)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"🎯 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for states, policy_targets, value_targets in tqdm(train_loader, desc="Training"):
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_pred, policy_targets)
            value_loss = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
            for states, policy_targets, value_targets in tqdm(val_loader, desc="Validation"):
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Calculate losses
                policy_loss = self.policy_criterion(policy_pred, policy_targets)
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
    
    def train(self, num_epochs=200, batch_size=64, save_path='watersort_imitation.pth'):
        """Main training loop"""
        print("🎯 Starting Imitation Learning Training...")
        
        # Load data
        train_dataset, val_dataset = self.load_data()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        
        print(f"📈 Training for {num_epochs} epochs...")
        print(f"📦 Batch size: {batch_size}")
        print(f"💾 Save path: {save_path}")
        print("-" * 70)
        
        for epoch in range(num_epochs):
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
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"Train: {train_loss:7.4f} (P: {train_policy_loss:6.4f}, V: {train_value_loss:6.4f}) | "
                  f"Val: {val_loss:7.4f} (P: {val_policy_loss:6.4f}, V: {val_value_loss:6.4f})")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch': epoch
                }, save_path)
                print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        print("-" * 70)
        print("✅ Training completed!")
        
        # Load best model for final evaluation
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model
    
    def evaluate_model(self, num_games=100):
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
                
                # Convert state to tensor
                state_tensor = self.processor.state_to_tensor(state).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    policy, value = self.model(state_tensor)
                
                # Get valid moves
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                
                # Convert policy to probabilities
                policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                
                # Filter valid moves and select best
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

# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================

def train_imitation_learning():
    """Main training function for imitation learning"""
    print("=" * 70)
    print("🎯 PHASE 2: IMITATION LEARNING TRAINING")
    print("=" * 70)
    
    # Initialize trainer
    trainer = ImitationTrainer(num_bottles=8, bottle_height=4, num_colors=6)
    
    # Train model
    model = trainer.train(
        num_epochs=200,
        batch_size=64,
        save_path='watersort_imitation.pth'
    )
    
    # Evaluate model
    print("\n" + "=" * 50)
    print("📊 FINAL EVALUATION")
    print("=" * 50)
    
    win_rate, avg_moves = trainer.evaluate_model(num_games=100)
    
    print(f"\n{'='*70}")
    print("✅ PHASE 2 COMPLETED!")
    print(f"   Model saved: watersort_imitation.pth")
    print(f"   Win rate: {win_rate:.1%}")
    print(f"   Avg moves: {avg_moves:.1f}")
    print(f"{'='*70}\n")
    
    return win_rate, avg_moves

if __name__ == "__main__":
    train_imitation_learning()