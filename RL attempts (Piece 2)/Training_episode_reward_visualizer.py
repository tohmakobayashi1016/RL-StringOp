import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from stable_baselines3 import DQN

# Path to the episode log file
log_file = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\episode_rewards_log.csv'
model = DQN.load(r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\dqn_mesh_graph_disregard')

# Load the CSV file
data = pd.read_csv(log_file)

# Add an 'episode_number' column based on the row index
data['episode_number'] = data.index + 1  # Starting episode numbers from 1

# Print the first few rows to inspect the data
print(data.head())

# Create a figure with subplots
plt.figure(figsize=(12, 8))

# Plot episode rewards over time
plt.subplot(2, 1, 1)
plt.plot(data['episode_number'], data['episode_reward'], label='Reward per Episode')
plt.title('Rewards Collected per Episode')
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# Smoothing reward plot using rolling mean for clearer trends (optional)
window_size = 10
plt.plot(data['episode_number'], data['episode_reward'].rolling(window=window_size).mean(), 
         label=f'Rolling Mean (window={window_size})', color='orange')
plt.legend()

# Plot episode lengths over time
plt.subplot(2, 1, 2)
plt.plot(data['episode_number'], data['episode_length'], label='Episode Length', color='green')
plt.title('Episode Length Over Time')
plt.xlabel('Episode Number')
plt.ylabel('Episode Length (Steps)')
plt.grid(True)
plt.legend()

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()
