import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from stable_baselines3 import DQN

# Path to the log file
log_file = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\training_log_disregard.csv'
model    = DQN.load(r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\dqn_mesh_graph_disregard')

q_network = model.q_net #Access the Q-network from the model
for layer in q_network.parameters(): #Inspect the weights of the Q-networks
    print(layer.data) #Visualize weights or biases for each layer

replay_buffer = model.replay_buffer
obs = replay_buffer.observations
actions = replay_buffer.actions
rewards = replay_buffer.rewards

obs_input = {
    'vertices': torch.tensor(obs['vertices'], dtype=torch.float32),
    'edge_attr': torch.tensor(obs['edge_attr'], dtype=torch.float32),
    'edge_index': torch.tensor(obs['edge_index'], dtype=torch.float32),
    'faces': torch.tensor(obs['faces'], dtype=torch.float32),
    'degree_histogram': torch.tensor(obs.get('degree_histogram', torch.zeros(1)), dtype=torch.float32),
    'levenshtein_distance': torch.tensor(obs.get('levenshtein_distance', torch.zeros(1)), dtype=torch.float32),
    'mesh_distance': torch.tensor(obs.get('mesh_distance', torch.zeros(1)), dtype=torch.float32)
}

# Now pass the dictionary to the q_net
q_values = model.q_net(obs_input)

# Read the log file
data = pd.read_csv(log_file)
print(data.head())

# Convert 'actions' from strings to lists of individual characters
data['actions'] = data['actions'].apply(lambda x: list(x))

# Create a figure
plt.figure(figsize=(18, 12))

# Plot rewards
plt.subplot(3, 2, 1)
plt.plot(data['reward'], label='Reward per Episode')
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# Smoothing reward plot using rolling mean for clearer trends
window_size = 10
plt.plot(data['reward'].rolling(window=window_size).mean(), label=f'Rolling Mean (window={window_size})', color='orange')
plt.legend()

# Plot episode lengths
plt.subplot(3, 2, 2)
plt.plot(data['length'], label='Length per Episode', color='green')
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.grid(True)
plt.legend()

# Plot action frequency heatmap
plt.subplot(3, 2, 3)
all_actions = pd.Series([action for sublist in data['actions'] for action in sublist])
action_counts = all_actions.value_counts().sort_index()  # Sort by action for better visualization

# Create a bar plot of action counts
sns.barplot(x=action_counts.index, y=action_counts.values, hue=action_counts.index, palette='viridis', legend=False)
plt.title('Action Frequency')
plt.xlabel('Action')
plt.ylabel('Count')

# Plot Levenshtein and Mesh distances instead of action sequences
plt.subplot(3, 2, 4)
plt.plot(data['current_mse'], label='Current MSE', color='blue')
plt.plot(data['distance_reward'], label='Distance Reward', color='orange')
plt.plot(data['time_step_penalty'], label='Time Step Penalty', color='red')
plt.title('Reward Components Over Time')
plt.xlabel('Episode')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

# Example: plotting rewards from the replay buffer
plt.subplot(3,2,5)
plt.plot(rewards)
plt.title('Rewards from Replay Buffer')
plt.xlabel('Timestep')
plt.ylabel('Reward')

# Visualize the Q-values for each action
plt.subplot(3,2,6)
plt.plot(q_values.detach().numpy())
plt.title('Q-values for Different Actions')
plt.xlabel('Observation Index')
plt.ylabel('Q-value')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
