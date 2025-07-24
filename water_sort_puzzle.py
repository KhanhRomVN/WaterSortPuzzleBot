!pip install torch numpy

import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class WaterSortEnv:
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128)   # Purple
    ]
    
    def __init__(self, num_tubes=5, tube_capacity=4):
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.reset()
        
    def reset(self):
        self.tubes = [[] for _ in range(self.num_tubes)]
        self.game_over = False
        self.moves = 0
        self.max_moves = 50
        
        # Generate random puzzle
        colors = random.sample(self.COLORS, self.num_tubes - 2)
        color_pool = colors * self.tube_capacity
        random.shuffle(color_pool)
        
        for i in range(self.num_tubes - 2):
            self.tubes[i] = color_pool[i*self.tube_capacity:(i+1)*self.tube_capacity]
        
        return self._get_state()
    
    def _get_state(self):
        """Convert game state to 3D tensor (height, width, channels)"""
        state = np.zeros((self.num_tubes, self.tube_capacity, 3), dtype=np.uint8)
        
        for tube_idx, tube in enumerate(self.tubes):
            for layer_idx, color in enumerate(tube):
                state[tube_idx, layer_idx] = color
                
        return state
    
    def is_valid_move(self, from_tube, to_tube):
        if from_tube == to_tube: 
            return False
        if not self.tubes[from_tube]: 
            return False
        if len(self.tubes[to_tube]) >= self.tube_capacity: 
            return False
            
        # Can pour if target tube is empty or top colors match
        if not self.tubes[to_tube]:
            return True
            
        return self.tubes[from_tube][-1] == self.tubes[to_tube][-1]
    
    def step(self, action):
        from_tube, to_tube = action
        reward = -0.1  # Small penalty for each move
        self.moves += 1
        
        if not self.is_valid_move(from_tube, to_tube):
            return self._get_state(), -1, self.game_over, {}
        
        # Move water
        color = self.tubes[from_tube].pop()
        self.tubes[to_tube].append(color)
        
        # Check win condition
        if self.check_win():
            reward = 10.0
            self.game_over = True
        
        # Check lose condition
        if self.moves >= self.max_moves:
            reward = -5.0
            self.game_over = True
            
        return self._get_state(), reward, self.game_over, {}
    
    def check_win(self):
        for tube in self.tubes:
            if not tube: 
                continue
            if len(tube) < self.tube_capacity: 
                return False
            if len(set(tube)) > 1: 
                return False
        return True

    def render(self):
        # Simple text-based rendering for testing
        print("\nCurrent State:")
        for i, tube in enumerate(self.tubes):
            print(f"Tube {i}: {tube}")
        print(f"Moves: {self.moves}\n")

# Test the environment
def test_env():
    print("Testing Water Sort Environment...")
    env = WaterSortEnv(num_tubes=5, tube_capacity=4)
    
    # Test 1: Initial state
    state = env.reset()
    print("Test 1: Initial state - PASSED" if len(state) == 5 else "Test 1: FAILED")
    
    # Test 2: Valid moves
    env.tubes = [[(255,0,0)], [], [(255,0,0)]]
    valid = env.is_valid_move(0, 1)
    print("Test 2: Valid move - PASSED" if valid else "Test 2: FAILED")
    
    # Test 3: Invalid moves
    env.tubes = [[(255,0,0)], [(0,255,0)], []]
    invalid = env.is_valid_move(0, 1)
    print("Test 3: Invalid move - PASSED" if not invalid else "Test 3: FAILED")
    
    # Test 4: Win condition
    env.tubes = [[(255,0,0)]*4, [(0,255,0)]*4, []]
    print("Test 4: Win condition - PASSED" if env.check_win() else "Test 4: FAILED")
    
    # Test 5: Step function
    _, reward, done, _ = env.step((0, 2))
    print("Test 5: Step execution - PASSED" if reward == -0.1 and not done else "Test 5: FAILED")

test_env()

class DQN(nn.Module):
    def __init__(self, num_tubes, tube_capacity):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, num_tubes, tube_capacity)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            conv_out_size = dummy.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_tubes * num_tubes)  # All possible moves

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class WaterSortAgent:
    def __init__(self, num_tubes, tube_capacity):
        self.num_tubes = num_tubes
        self.tube_capacity = tube_capacity
        self.policy_net = DQN(num_tubes, tube_capacity)
        self.target_net = DQN(num_tubes, tube_capacity)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            # Random valid action
            possible_moves = [(i, j) for i in range(self.num_tubes) 
                             for j in range(self.num_tubes) if i != j]
            return random.choice(possible_moves)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                q_values = q_values.view(self.num_tubes, self.num_tubes)
                
                # Mask invalid moves
                for i in range(self.num_tubes):
                    for j in range(self.num_tubes):
                        if i == j:
                            q_values[i, j] = -float('inf')
                
                # Select best valid move
                action_idx = torch.argmax(q_values).item()
                from_tube = action_idx // self.num_tubes
                to_tube = action_idx % self.num_tubes
                return (from_tube, to_tube)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Convert actions to indices
        action_indices = [from_idx * self.num_tubes + to_idx 
                         for (from_idx, to_idx) in actions]
        
        # Current Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, torch.tensor(action_indices).unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training function
def train_agent(episodes=1000):
    env = WaterSortEnv(num_tubes=5, tube_capacity=4)
    agent = WaterSortAgent(num_tubes=5, tube_capacity=4)
    
    scores = []
    losses = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Train on batch
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)
        
        # Update target network
        if episode % 10 == 0:
            agent.update_target_net()
        
        scores.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}: Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title("Episode Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    
    # Save model
    torch.save(agent.policy_net.state_dict(), "water_sort_model.pth")
    return agent

# Uncomment to start training
# trained_agent = train_agent(episodes=500)