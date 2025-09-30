# Water Sort Puzzle Bot 🧪

A sophisticated reinforcement learning agent that solves Water Sort Puzzle games using Proximal Policy Optimization (PPO) with Graph Neural Networks (GNN) and Transformer architectures.

## Overview

This project implements an AI agent that learns to solve the popular Water Sort Puzzle game through deep reinforcement learning. The agent combines:

- **Graph Neural Networks (GNN)** for spatial reasoning about tube connections
- **Transformer encoders** for sequence modeling of liquid colors
- **Proximal Policy Optimization (PPO)** for stable training

## Features

- 🧠 **Advanced Architecture**: GNN + Transformer for optimal state representation
- 🎯 **Curriculum Learning**: Progressive difficulty across multiple stages
- 💾 **Memory Optimized**: Efficient GPU memory management with adaptive batch sizes
- 📊 **Comprehensive Monitoring**: Real-time resource usage and training metrics
- 🔄 **Robust Training**: Gradient clipping, KL divergence monitoring, and OOM protection

## Installation

```bash
# Clone the repository
git clone https://github.com/KhanhRomVN/WaterSortPuzzleBot.git
cd WaterSortPuzzleBot

# Install dependencies
pip install torch numpy torch_geometric GPUtil psutil
```

````

## Usage

```python
# Run the training pipeline
python water_sort_puzzle.py
```

The training process automatically progresses through multiple difficulty stages:

1. **Stage 1**: 5 tubes, 3 colors, 4 capacity
2. **Stage 2**: 8 tubes, 5 colors, 4 capacity
3. **Stage 3**: 12 tubes, 8 colors, 5 capacity

## Model Architecture

### WaterSortGNN

- **Input**: Tube state representation (completion flag + color sequences)
- **GNN Layers**: Graph Isomorphism Networks for tube connectivity
- **Transformer**: Self-attention for global context understanding
- **Output**: Action probabilities for all valid tube-to-tube transfers

### Training Strategy

- **PPO Algorithm**: Stable policy optimization with clipped objectives
- **Adaptive Batching**: Dynamic batch size adjustment for memory efficiency
- **Multi-stage Curriculum**: Progressive complexity for better learning

## Results

The agent achieves:

- High win rates across all difficulty levels
- Efficient solution paths with minimal moves
- Robust performance on unseen puzzle configurations

## Monitoring

Real-time tracking of:

- Training loss and reward curves
- Gradient norms and KL divergence
- CPU/GPU memory usage
- Training time per episode

## File Structure

```
WaterSortPuzzleBot/
├── water_sort_puzzle.py  # Main training script
└── README.md
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torch_geometric
- numpy
- GPUtil
- psutil

## Author

**KhanhRomVN**

- GitHub: [KhanhRomVN](https://github.com/KhanhRomVN)
- Email: khanhromvn@gmail.com

## License

This project is open source and available under the [MIT License](LICENSE).
````
