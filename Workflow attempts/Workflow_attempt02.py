import os, compas
from time import time
from math import pi, cos, sin
from collections import Counter, namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import add_strip_lizard
import compas_quad.grammar.lizard
from compas_fd.solvers import fd_numpy
from compas_viewer.viewer import Viewer

#Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(12, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#Hyperparameters
num_episodes = 1000
max_steps_per_episode = 100
batch_size = 32
gamma = 0.99 #Discount rate
epsilon = 1.0 #Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
learning_rate = 0.001

#Environment parameters
grid_size = (10,10)
num_actions = 4 #Turn, pivot, add, delete

input_dim = grid_size[0]*grid_size[1]

#Initizatize the environment, Q-network, target network, and replay memory
q_network = QNetwork(input_dim, num_actions)
target_network = QNetwork(input_dim, num_actions)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

replay_memory = deque(maxlen=2000)

#Helper functions
def get_state(grid):
    return grid.flatten()

def get_next_action(state, epsilon, num_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)
    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = q_network(state)
    return torch.argmax(q_values).item()

def apply_action(grid, marker, action):
    if action == 0:  # Up
        marker.move_up(grid_size)
    elif action == 1:  # Down
        marker.move_down(grid_size)
    elif action == 2:  # Left
        marker.move_left(grid_size)
    elif action == 3:  # Right
        marker.move_right(grid_size)
    elif action == 4:  # Add object using add_object
        grid = add_object(grid, marker.position)
    elif action == 5:  # Add object using add_object2
        grid = add_object2(grid, marker.position)
    elif action == 6:  # Remove object
        grid = delete_object(grid, marker.position)
    elif action == 7:  # Custom action
        grid = custom_action(grid, marker.position)
    return grid

# Define a custom action (for example purposes)
def custom_action(grid, position):
    # Implement your custom grammar rule here
    return grid

# Training the DQN
for episode in range(num_episodes):
    grid = np.zeros(grid_size)
    marker = Lizard((0, 0))
    state = get_state(grid)
    
    for step in range(max_steps_per_episode):
        action = get_next_action(state, epsilon, num_actions)
        next_grid = apply_action(grid, marker, action)
        next_state = get_state(next_grid)
        reward = 1 if (marker.position == (grid_size[0]-1, grid_size[1]-1)) else -1
        done = marker.position == (grid_size[0]-1, grid_size[1]-1)
        
        replay_memory.append((state, action, reward, next_state, done))
        state = next_state
        grid = next_grid
        
        if done:
            break
        
        if len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatFloat(next_states)
            dones = torch.FloatTensor(dones)
            
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            max_next_q_values = target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * gamma * max_next_q_values
            
            loss = criterion(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epsilon > min_epsilon:
                epsilon *= epsilon_decay
        
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())
        print(f'Episode {episode}, Epsilon: {epsilon}')

print("Training finished.")