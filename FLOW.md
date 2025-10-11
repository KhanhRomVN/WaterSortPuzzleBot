# Flow Training - Water Sort Puzzle AI

## 📋 Tổng quan
Project sử dụng **4-phase training**:
1. **Phase 1 (heuristic_solver.py)**: Tạo dữ liệu huấn luyện từ heuristic solver
2. **Phase 2 (imitation_learning.py)**: Huấn luyện mô hình với Imitation Learning
3. **Phase 3 (self_play.py)**: Tạo dữ liệu self-play với MCTS
4. **Phase 4 (reinforcement_learning.py)**: Huấn luyện mô hình với dữ liệu self-play (AlphaZero)

---

## 🔄 PHASE 1: HEURISTIC SOLVER (heuristic_solver.py)

### 1. Heuristic Solver
- Thuật toán: **Greedy Best-First Search** với 8 heuristics
- Tỷ lệ giải: **???%**
- Tạo expert demonstrations

### 2. Generate Training Data
- **5,000 games** (10 iterations × 500 games)
- **~50,000-100,000 samples** (state, action, value)
- Mỗi sample: `(state, policy_target, value_target)`

### 3. Output
- File: `heuristic_data.pkl` (chứa các samples)

---

## 🚀 PHASE 2: IMITATION LEARNING (imitation_learning.py)

### 4. Load Training Data
- Load `heuristic_data.pkl` từ `PHASE 1`

### 5. Train Neural Network
- Thuật toán: **Imitation Learning** (Behavioral Cloning)
- Loss: **Cross-entropy** (policy) + **MSE** (value)
- **200 epochs** (10 iter × 20 epochs)
- Optimizer: **Adam** (lr=0.001, weight_decay=1e-4)

### 6. Output
- File: `watersort_imitation.pth`
- Win rate: **??-??%**
- Training time: **30-45 phút**

---

## 🔁 PHASE 3: SELF-PLAY (self_play.py)

### 7. Load Pre-trained Model
- Load `watersort_imitation.pth` từ `PHASE 2`

### 8. Self-Play Loop
- **100 games** per iteration
- Mỗi game: MCTS search (**100 simulations**/move)
- Generate trajectories: `(state, π, z)`
- Store vào **replay buffer** (max 10,000 samples)

### 9. Output
- File: `self_play_data.pkl` (chứa các trajectories)

---

## 🎯 PHASE 4: REINFORCEMENT LEARNING (reinforcement_learning.py)

### 10. Load Self-Play Data
- Load `self_play_data.pkl` từ `PHASE 3`

### 11. Train with Self-Play Data
- Loss: **KL divergence** (policy) + **MSE** (value)
- **10 epochs** per iteration
- Batch size: **32**
- Optimizer: **Adam** (lr=0.001 hoặc 0.0001 nếu fine-tune)

### 12. Evaluation
- Test mỗi **5 iterations**
- Đo win rate và average moves
- Save best model

### 13. Output
- File: `watersort_model.pth`
- Win rate: **90-100%**
- Training time: **10-20 giờ** (100 iterations)

---

**Author**: KhanhRomVN  
**GitHub**: [WaterSortPuzzleBot](https://github.com/KhanhRomVN/WaterSortPuzzleBot)