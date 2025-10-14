# imitation_learning_optimized.py
# Phase 2: Train model with Imitation Learning - OPTIMIZED VERSION
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
import time
import os
import glob
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. COLAB FILE PICKER
# =============================================================================

def select_pkl_file_colab():
    """Chọn file pkl trong Colab environment"""
    pkl_files = glob.glob('*.pkl')
    
    if not pkl_files:
        print("❌ Không tìm thấy file .pkl nào trong thư mục hiện tại")
        print("📁 Vui lòng upload file .pkl trước!")
        return None
    
    print("📁 CÁC FILE PKL CÓ SẴN:")
    print("-" * 70)
    for idx, filename in enumerate(pkl_files, 1):
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        print(f"  {idx}. {filename:<40} ({file_size:.1f} MB)")
    
    print("-" * 70)
    while True:
        try:
            choice = int(input("Nhập số thứ tự file (ví dụ: 1): "))
            if 1 <= choice <= len(pkl_files):
                selected_file = pkl_files[choice - 1]
                print(f"✅ Đã chọn: {selected_file}\n")
                return selected_file
            else:
                print(f"❌ Vui lòng nhập số từ 1 đến {len(pkl_files)}")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ")

# =============================================================================
# 1. NEURAL NETWORK ARCHITECTURE - OPTIMIZED
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
        
        # Policy head - cải thiện
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.policy_fc1 = nn.Linear(64 * num_bottles * bottle_height, 512)
        self.policy_fc2 = nn.Linear(512, num_bottles * num_bottles)
        self.policy_bn = nn.BatchNorm1d(512)
        
        # Value head - cải thiện
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
# 2. DATA PREPROCESSING - OPTIMIZED
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
    
    def process_dataset_lazy(self, dataset):
        """Load dataset lớn mà không cần load toàn bộ vào memory"""
        print("🔄 Processing dataset (lazy loading)...")
        
        states = []
        policy_targets = []
        value_targets = []
        
        for state, policy_target, value_target in tqdm(dataset, desc="Converting"):
            state_tensor = self.state_to_tensor(state)
            policy_tensor = torch.from_numpy(policy_target).float()
            value_tensor = torch.tensor([value_target], dtype=torch.float32)
            
            states.append(state_tensor)
            policy_targets.append(policy_tensor)
            value_targets.append(value_tensor)
        
        return (torch.stack(states), 
                torch.stack(policy_targets), 
                torch.stack(value_targets))

# =============================================================================
# 3. IMITATION LEARNING TRAINER - OPTIMIZED
# =============================================================================

class ImitationTrainer:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = WaterSortNet(num_bottles, bottle_height, num_colors).to(self.device)
        self.processor = DataProcessor(num_bottles, bottle_height, num_colors)
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning rate scheduler - CỨU TỐI ƯU MỚI
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-5
        )
        
        # Gradient clipping - CỨU TỐI ƯU MỚI
        self.grad_clip = 1.0
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.learning_rates = []
        
        # Early stopping - CỨU TỐI ƯU MỚI
        self.patience = 15
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def load_data(self, filename):
        """Load và process training data"""
        print(f"📁 Loading data từ {filename}...")
        
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"📊 Loaded {len(dataset)} samples")
        
        # Process data
        states, policy_targets, value_targets = self.processor.process_dataset_lazy(dataset)
        
        # Normalize value targets
        value_targets = (value_targets - value_targets.mean()) / (value_targets.std() + 1e-8)
        
        # Create dataset
        full_dataset = TensorDataset(states, policy_targets, value_targets)
        
        # Split train/validation (90/10)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"🎯 Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train một epoch"""
        self.model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for states, policy_targets, value_targets in pbar:
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_criterion(policy_pred, policy_targets)
            value_loss = self.value_criterion(value_pred.squeeze(-1), value_targets.squeeze(-1))
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - CỨU TỐI ƯU MỚI
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'p_loss': f'{policy_loss.item():.4f}',
                'v_loss': f'{value_loss.item():.4f}'
            })
        
        return (total_loss / num_batches,
                total_policy_loss / num_batches,
                total_value_loss / num_batches)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, policy_targets, value_targets in tqdm(val_loader, desc="Validation", leave=False):
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Calculate losses
                policy_loss = self.policy_criterion(policy_pred, policy_targets)
                value_loss = self.value_criterion(value_pred.squeeze(-1), value_targets.squeeze(-1))
                
                # Combined loss
                loss = policy_loss + value_loss
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
        
        return (total_loss / num_batches,
                total_policy_loss / num_batches,
                total_value_loss / num_batches)
    
    def train(self, pkl_filename, num_epochs=200, batch_size=32, save_path='watersort_imitation.pth'):
        """Main training loop"""
        print("=" * 70)
        print("🎯 PHASE 2: IMITATION LEARNING - OPTIMIZED")
        print("=" * 70 + "\n")
        
        # Load data
        train_dataset, val_dataset = self.load_data(pkl_filename)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2, 
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=2
        )
        
        print(f"📈 Config:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate schedule: CosineAnnealing (1e-3 → 1e-5)")
        print(f"   Gradient clip: {self.grad_clip}")
        print(f"   Early stopping: {self.patience} epochs")
        print(f"   Save path: {save_path}")
        print("-" * 70 + "\n")
        
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_p_loss, train_v_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_p_loss, val_v_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.policy_losses.append(train_p_loss)
            self.value_losses.append(train_v_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            self.scheduler.step()
            
            # Print progress
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"LR: {lr:.2e} | "
                  f"Train: {train_loss:7.4f} (P: {train_p_loss:6.4f}, V: {train_v_loss:6.4f}) | "
                  f"Val: {val_loss:7.4f} (P: {val_p_loss:6.4f}, V: {val_v_loss:6.4f})")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch': epoch
                }, save_path)
                print(f"   💾 Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                    break
        
        total_elapsed = time.time() - total_start_time
        
        print("-" * 70)
        print(f"✅ Training completed in {str(timedelta(seconds=int(total_elapsed)))}")
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model
    
    def evaluate_model(self, num_games=100):
        """Evaluate model performance (basic version)"""
        print("\n" + "=" * 70)
        print("🧪 MODEL EVALUATION")
        print("=" * 70)
        
        # Import environment từ heuristic_solver
        try:
            from heuristic_solver import WaterSortEnv
        except ImportError:
            print("⚠️  Cannot import WaterSortEnv - skipping evaluation")
            print("   Make sure heuristic_solver.py is in the same directory")
            return None, None
        
        print(f"\n📊 Evaluating on {num_games} games...\n")
        
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
                state, _, _, _ = env.step(best_move)
                moves += 1
        
        win_rate = wins / num_games
        avg_moves = total_moves / max(wins, 1)
        
        print(f"\n✅ Win rate: {wins}/{num_games} ({win_rate:.1%})")
        if wins > 0:
            print(f"📈 Avg moves (when win): {avg_moves:.1f}")
        
        return win_rate, avg_moves

# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================

def train_imitation_learning():
    """Main training function for imitation learning"""
    # Select file in Colab
    pkl_file = select_pkl_file_colab()
    
    if pkl_file is None:
        return
    
    # Initialize trainer
    trainer = ImitationTrainer(num_bottles=8, bottle_height=4, num_colors=6)
    
    # Train model
    model = trainer.train(
        pkl_filename=pkl_file,
        num_epochs=200,
        batch_size=32,
        save_path='watersort_imitation.pth'
    )
    
    # Evaluate model
    win_rate, avg_moves = trainer.evaluate_model(num_games=100)
    
    # Final report
    print("\n" + "=" * 70)
    print("✅ PHASE 2 COMPLETED!")
    print("=" * 70)
    print(f"📁 Loaded from: {pkl_file}")
    print(f"💾 Model saved: watersort_imitation.pth")
    if win_rate is not None:
        print(f"🎯 Win rate: {win_rate:.1%}")
        print(f"📈 Avg moves: {avg_moves:.1f}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    train_imitation_learning()