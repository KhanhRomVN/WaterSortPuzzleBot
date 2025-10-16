# reinforcement_learning.py - OPTIMIZED VERSION
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
import glob
import matplotlib.pyplot as plt
from collections import deque
import gc
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. FILE PICKER FOR COLAB
# =============================================================================

def select_pkl_file_colab():
    """Chọn file .pkl trong Colab environment"""
    pkl_files = glob.glob('*.pkl')
    
    if not pkl_files:
        print("❌ Không tìm thấy file .pkl nào trong thư mục hiện tại")
        print("📁 Vui lòng upload file self-play data trước!")
        return None
    
    print("📁 CÁC FILE PKL CÓ SẴN:")
    print("-" * 70)
    for idx, filename in enumerate(pkl_files, 1):
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        print(f"  {idx}. {filename:<40} ({file_size:.1f} MB)")
    
    print("-" * 70)
    while True:
        try:
            choice = int(input("Nhập số thứ tự file self-play data (ví dụ: 1): "))
            if 1 <= choice <= len(pkl_files):
                selected_file = pkl_files[choice - 1]
                print(f"✅ Đã chọn: {selected_file}\n")
                return selected_file
            else:
                print(f"❌ Vui lòng nhập số từ 1 đến {len(pkl_files)}")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ")

def select_pth_file_colab():
    """Chọn file .pth trong Colab environment"""
    pth_files = glob.glob('*.pth')
    
    if not pth_files:
        print("⚠️  Không tìm thấy file .pth nào, sẽ train từ đầu")
        return None
    
    print("📁 CÁC FILE PTH CÓ SẴN:")
    print("-" * 70)
    for idx, filename in enumerate(pth_files, 1):
        file_size = os.path.getsize(filename) / (1024**2)  # MB
        print(f"  {idx}. {filename:<40} ({file_size:.1f} MB)")
    
    print("  0. Train từ đầu (random initialization)")
    print("-" * 70)
    
    while True:
        try:
            choice = input("Nhập số thứ tự file model (0 để train từ đầu): ")
            if choice == '0':
                print("✅ Sẽ train từ đầu\n")
                return None
            choice = int(choice)
            if 1 <= choice <= len(pth_files):
                selected_file = pth_files[choice - 1]
                print(f"✅ Đã chọn: {selected_file}\n")
                return selected_file
            else:
                print(f"❌ Vui lòng nhập số từ 0 đến {len(pth_files)}")
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
# 4. OPTIMIZED REINFORCEMENT LEARNING TRAINER
# =============================================================================

class RLTrainer:
    def __init__(self, num_bottles=8, bottle_height=4, num_colors=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize model
        self.num_bottles = num_bottles
        self.bottle_height = bottle_height
        self.num_colors = num_colors
        
        self.model = WaterSortNet(num_bottles, bottle_height, num_colors).to(self.device)
        self.processor = DataProcessor(num_bottles, bottle_height, num_colors)
        
        # KHỞI TẠO OPTIMIZER TRƯỚC
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Load pre-trained model if available
        self._load_pretrained_model()
        
        # Loss functions - FIXED: Use CrossEntropy for policy, MSE for value
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Scheduler - FIXED: Remove 'verbose' parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.policy_losses = []
        self.value_losses = []
        self.win_rates = []
        self.best_win_rate = 0.0
        
        # Replay buffer với memory management
        self.replay_buffer = deque(maxlen=10000)
        
        print("✅ RL Trainer initialized successfully")
    
    def _load_pretrained_model(self):
        """Load pre-trained model từ imitation learning"""
        model_path = select_pth_file_colab()
        
        if model_path and os.path.exists(model_path):
            print(f"📁 Loading pre-trained model from {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Pre-trained model loaded successfully")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("✅ Model weights loaded successfully")
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("✅ Optimizer state loaded")
                    
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Continuing with random initialization")
        else:
            print("🔄 No pre-trained model found, starting from scratch")
    
    def load_self_play_data(self, filename=None):
        """Load and process self-play data"""
        if filename is None:
            filename = select_pkl_file_colab()
            if filename is None:
                return 0
        
        print(f"📁 Loading self-play data from {filename}...")
        try:
            with open(filename, 'rb') as f:
                self_play_data = pickle.load(f)
            
            print(f"📊 Loaded {len(self_play_data)} self-play samples")
            
            # Add to replay buffer với memory management
            added_count = 0
            for sample in self_play_data:
                if len(self.replay_buffer) >= self.replay_buffer.maxlen:
                    break
                self.replay_buffer.append(sample)
                added_count += 1
            
            print(f"📥 Added {added_count} samples to replay buffer")
            print(f"📦 Replay buffer size: {len(self.replay_buffer)}")
            
            return added_count
            
        except Exception as e:
            print(f"❌ Error loading self-play data: {e}")
            return 0
    
    def prepare_training_data(self, batch_size=32, validation_split=0.1):
        """Prepare training data từ replay buffer với memory optimization"""
        if len(self.replay_buffer) < batch_size:
            print(f"⚠️  Not enough samples in replay buffer ({len(self.replay_buffer)} < {batch_size})")
            return None, None
        
        # Sample ngẫu nhiên từ replay buffer
        num_samples = min(len(self.replay_buffer), batch_size * 20)  # Limit size
        samples = [self.replay_buffer[i] for i in np.random.choice(len(self.replay_buffer), num_samples, replace=False)]
        
        states = []
        policy_targets = []
        value_targets = []
        
        for state, policy_target, value_target in samples:
            try:
                state_tensor = self.processor.state_to_tensor(state)
                
                # Convert policy target to index for CrossEntropyLoss
                policy_index = np.argmax(policy_target)
                policy_tensor = torch.LongTensor([policy_index])
                
                value_tensor = torch.FloatTensor([value_target])
                
                states.append(state_tensor)
                policy_targets.append(policy_tensor)
                value_targets.append(value_tensor)
            except Exception as e:
                continue  # Skip corrupted samples
        
        if len(states) < batch_size:
            print(f"⚠️  Not enough valid samples after processing ({len(states)} < {batch_size})")
            return None, None
        
        # Create dataset
        dataset = TensorDataset(
            torch.stack(states), 
            torch.stack(policy_targets), 
            torch.stack(value_targets)
        )
        
        # Split train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        if val_size == 0:
            train_dataset = dataset
            val_dataset = dataset  # Use same dataset for validation if too small
        else:
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"🎯 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch với memory optimization"""
        self.model.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for states, policy_targets, value_targets in pbar:
            try:
                # Move to device
                states = states.to(self.device, non_blocking=True)
                policy_targets = policy_targets.to(self.device, non_blocking=True).squeeze()
                value_targets = value_targets.to(self.device, non_blocking=True).squeeze()
                
                # Forward pass
                policy_pred, value_pred = self.model(states)
                
                # Calculate losses - FIXED: Use CrossEntropy directly
                policy_loss = self.policy_criterion(policy_pred, policy_targets)
                value_loss = self.value_criterion(value_pred.squeeze(), value_targets)
                
                # Combined loss (AlphaZero style)
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad(set_to_none=True)  # Memory optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Policy': f'{policy_loss.item():.4f}', 
                    'Value': f'{value_loss.item():.4f}'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("💥 GPU out of memory, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
            
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
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            
            for states, policy_targets, value_targets in pbar:
                try:
                    # Move to device
                    states = states.to(self.device, non_blocking=True)
                    policy_targets = policy_targets.to(self.device, non_blocking=True).squeeze()
                    value_targets = value_targets.to(self.device, non_blocking=True).squeeze()
                    
                    # Forward pass
                    policy_pred, value_pred = self.model(states)
                    
                    # Calculate losses
                    policy_loss = self.policy_criterion(policy_pred, policy_targets)
                    value_loss = self.value_criterion(value_pred.squeeze(), value_targets)
                    
                    # Combined loss
                    loss = policy_loss + value_loss
                    
                    # Update statistics
                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Val Loss': f'{loss.item():.4f}',
                        'Policy': f'{policy_loss.item():.4f}',
                        'Value': f'{value_loss.item():.4f}'
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("💥 GPU out of memory during validation, skipping batch")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
            
        return (total_loss / num_batches, 
                total_policy_loss / num_batches, 
                total_value_loss / num_batches)
    
    def evaluate_model(self, num_games=50, max_moves=200):
        """Evaluate model performance - SIMPLIFIED VERSION"""
        print("🧪 Evaluating model...")
        
        self.model.eval()
        env = WaterSortEnv()
        wins = 0
        total_moves = 0
        move_counts = []
        
        pbar = tqdm(range(num_games), desc="Evaluation")
        
        for game_idx in pbar:
            state = env.reset()
            moves = 0
            game_won = False
            
            for move_idx in range(max_moves):
                if env.is_solved():
                    wins += 1
                    total_moves += moves
                    move_counts.append(moves)
                    game_won = True
                    break
                
                # Get valid moves
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    break
                
                # Use direct policy for evaluation (faster)
                state_tensor = self.processor.state_to_tensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    policy, value = self.model(state_tensor)
                
                policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
                
                # Select best valid move
                best_move = None
                best_score = -float('inf')
                
                for move in valid_moves:
                    from_idx, to_idx = move
                    move_index = from_idx * self.num_bottles + to_idx
                    score = policy_probs[move_index]
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
                
                if best_move is None:
                    break
                
                # Execute move
                state, reward, done, _ = env.step(best_move)
                moves += 1
            
            # Update progress
            current_win_rate = wins / (game_idx + 1)
            pbar.set_postfix({'Win Rate': f'{current_win_rate:.1%}'})
        
        win_rate = wins / num_games
        avg_moves = total_moves / max(wins, 1)
        
        print(f"🎯 Evaluation Results:")
        print(f"   Win rate: {wins}/{num_games} ({win_rate:.1%})")
        print(f"   Avg moves (when win): {avg_moves:.1f}")
        
        return win_rate, avg_moves
    
    def train(self, num_iterations=10, epochs_per_iter=5, batch_size=32, 
              eval_every=2, save_path='watersort_rl_model.pth'):
        """Main reinforcement learning training loop với memory management"""
        print("🎯 Starting Reinforcement Learning Training...")
        print(f"   Iterations: {num_iterations}")
        print(f"   Epochs per iteration: {epochs_per_iter}")
        print(f"   Batch size: {batch_size}")
        print(f"   Save path: {save_path}")
        print("-" * 70)
        
        # Load initial self-play data
        data_loaded = self.load_self_play_data()
        if data_loaded == 0:
            print("❌ No self-play data loaded. Cannot start training.")
            return 0.0, 0.0
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")
            
            # Clear memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Prepare training data
            train_dataset, val_dataset = self.prepare_training_data(batch_size)
            if train_dataset is None:
                print("❌ Not enough training data, skipping iteration")
                continue
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=0, pin_memory=True)  # num_workers=0 for stability
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=True)
            
            # Train for multiple epochs
            iteration_train_losses = []
            iteration_val_losses = []
            
            for epoch in range(epochs_per_iter):
                epoch_start = time.time()
                
                # Train
                train_loss, train_policy_loss, train_value_loss = self.train_epoch(train_loader)
                
                # Validate
                val_loss, val_policy_loss, val_value_loss = self.validate(val_loader)
                
                epoch_time = time.time() - epoch_start
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.policy_losses.append(train_policy_loss)
                self.value_losses.append(train_value_loss)
                
                iteration_train_losses.append(train_loss)
                iteration_val_losses.append(val_loss)
                
                # Print progress
                print(f"  Epoch {epoch+1:2d}/{epochs_per_iter} | "
                      f"Time: {epoch_time:5.1f}s | "
                      f"Train: {train_loss:7.4f} (P: {train_policy_loss:6.4f}, V: {train_value_loss:6.4f}) | "
                      f"Val: {val_loss:7.4f} (P: {val_policy_loss:6.4f}, V: {val_value_loss:6.4f})")
            
            # Step learning rate scheduler based on validation loss
            avg_val_loss = np.mean(iteration_val_losses)
            self.scheduler.step(avg_val_loss)
            
            # Evaluate model periodically
            if (iteration + 1) % eval_every == 0 or iteration == num_iterations - 1:
                print("\n📊 Running evaluation...")
                win_rate, avg_moves = self.evaluate_model(num_games=50)
                self.win_rates.append(win_rate)
                
                # Save best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self._save_model(save_path, iteration, win_rate)
                    print(f"💾 Saved best model (win_rate: {win_rate:.1%})")
                else:
                    # Save checkpoint anyway
                    checkpoint_path = f"checkpoint_iter_{iteration+1}.pth"
                    self._save_model(checkpoint_path, iteration, win_rate)
                    print(f"💾 Saved checkpoint: {checkpoint_path}")
            
            # Clean up memory
            del train_loader, val_loader, train_dataset, val_dataset
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        print("-" * 70)
        print(f"✅ Reinforcement Learning completed in {total_time:.1f}s!")
        
        # Final evaluation
        print("\n🎯 FINAL EVALUATION")
        final_win_rate, final_avg_moves = self.evaluate_model(num_games=100)
        
        # Save final model
        self._save_model(save_path, num_iterations, final_win_rate, final=True)
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   Final win rate: {final_win_rate:.1%}")
        print(f"   Best win rate: {self.best_win_rate:.1%}")
        print(f"   Total training time: {total_time:.1f}s")
        print(f"   Model saved: {save_path}")
        
        return final_win_rate, final_avg_moves
    
    def _save_model(self, save_path, iteration, win_rate, final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': iteration,
            'win_rate': win_rate,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'win_rates': self.win_rates,
            'config': {
                'num_bottles': self.num_bottles,
                'bottle_height': self.bottle_height,
                'num_colors': self.num_colors
            }
        }
        
        torch.save(checkpoint, save_path)
        
        file_size = os.path.getsize(save_path) / (1024**2)
        if final:
            print(f"💾 Saved final model to {save_path} ({file_size:.1f} MB)")
        else:
            print(f"💾 Saved model checkpoint (iter {iteration}, win_rate: {win_rate:.1%}, size: {file_size:.1f} MB)")
    
    def plot_training_progress(self, save_path='training_progress.png'):
        """Plot training progress"""
        if not self.train_losses:
            print("❌ No training data to plot")
            return
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot losses
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', alpha=0.7)
            ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', alpha=0.7)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot policy and value losses
            ax2.plot(epochs, self.policy_losses, 'g-', label='Policy Loss', alpha=0.7)
            ax2.plot(epochs, self.value_losses, 'm-', label='Value Loss', alpha=0.7)
            ax2.set_title('Policy and Value Losses')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot win rates
            if self.win_rates:
                eval_points = range(1, len(self.win_rates) + 1)
                ax3.plot(eval_points, self.win_rates, 'c-', marker='o', label='Win Rate', alpha=0.7)
                ax3.axhline(y=self.best_win_rate, color='r', linestyle='--', label=f'Best: {self.best_win_rate:.1%}')
                ax3.set_title('Win Rate Progress')
                ax3.set_xlabel('Evaluation Point')
                ax3.set_ylabel('Win Rate')
                ax3.set_ylim(0, 1)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Training info
            ax4.axis('off')
            info_text = (
                f"Training Information:\n"
                f"• Best Win Rate: {self.best_win_rate:.1%}\n"
                f"• Final LR: {self.optimizer.param_groups[0]['lr']:.2e}\n"
                f"• Total Epochs: {len(self.train_losses)}\n"
                f"• Device: {self.device}\n"
                f"• Model: {self.num_bottles}b_{self.bottle_height}h_{self.num_colors}c"
            )
            ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Training progress plot saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Error plotting training progress: {e}")

# =============================================================================
# 5. MAIN REINFORCEMENT LEARNING FUNCTION
# =============================================================================

def train_reinforcement_learning():
    """Main reinforcement learning training function"""
    print("=" * 70)
    print("🎯 PHASE 4: REINFORCEMENT LEARNING TRAINING (OPTIMIZED)")
    print("=" * 70)
    
    # Configuration
    print("⚙️  CẤU HÌNH REINFORCEMENT LEARNING:")
    print("-" * 70)
    
    # Số iterations
    while True:
        try:
            num_iterations = int(input("Nhập số iterations (mặc định: 10): ") or "10")
            if num_iterations > 0:
                break
            else:
                print("❌ Số iterations phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Epochs per iteration
    while True:
        try:
            epochs_per_iter = int(input("Nhập số epochs per iteration (mặc định: 5): ") or "5")
            if epochs_per_iter > 0:
                break
            else:
                print("❌ Số epochs phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Batch size
    while True:
        try:
            batch_size = int(input("Nhập batch size (mặc định: 32): ") or "32")
            if batch_size > 0:
                break
            else:
                print("❌ Batch size phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    # Evaluation frequency
    while True:
        try:
            eval_every = int(input("Nhập evaluation frequency (mặc định: 2): ") or "2")
            if eval_every > 0:
                break
            else:
                print("❌ Evaluation frequency phải lớn hơn 0!")
        except ValueError:
            print("❌ Vui lòng nhập số nguyên hợp lệ!")
    
    print("-" * 70)
    print(f"✅ Configuration:")
    print(f"   Iterations: {num_iterations}")
    print(f"   Epochs per iteration: {epochs_per_iter}")
    print(f"   Batch size: {batch_size}")
    print(f"   Evaluate every: {eval_every} iterations")
    print("=" * 70 + "\n")
    
    # Initialize trainer
    trainer = RLTrainer(num_bottles=8, bottle_height=4, num_colors=6)
    
    # Train model with reinforcement learning
    final_win_rate, final_avg_moves = trainer.train(
        num_iterations=num_iterations,
        epochs_per_iter=epochs_per_iter,
        batch_size=batch_size,
        eval_every=eval_every,
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
# 6. MODEL TESTING FUNCTION
# =============================================================================

def test_rl_model(model_path='watersort_rl_model.pth', num_games=100):
    """Test trained RL model"""
    print("=" * 70)
    print("🧪 RL MODEL TESTING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaterSortNet(num_bottles=8, bottle_height=4, num_colors=6).to(device)
    processor = DataProcessor()
    
    # Load trained model
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from {model_path}")
            if 'win_rate' in checkpoint:
                print(f"   Model win rate during training: {checkpoint['win_rate']:.1%}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    else:
        print(f"❌ Model file {model_path} not found")
        return
    
    model.eval()
    
    # Test với direct policy
    print(f"\n🔍 Testing with direct policy...")
    
    env = WaterSortEnv()
    wins = 0
    total_moves = 0
    move_counts = []
    
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
    print(f"   Method: Direct Policy")
    
    return win_rate, avg_moves

if __name__ == "__main__":
    # Run reinforcement learning training
    final_win_rate, final_avg_moves = train_reinforcement_learning()
    
    # Test the final model
    print("\n" + "="*70)
    print("FINAL RL MODEL TESTING")
    print("="*70)
    
    test_rl_model('watersort_rl_model.pth', num_games=100)