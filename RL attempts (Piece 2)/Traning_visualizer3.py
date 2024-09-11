import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from stable_baselines3 import DQN

# Path to the log file
log_file = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\training_log_exp_ver.csv'
model    = DQN.load(r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\dqn_mesh_graph_exp_ver')

q_network = model.q_net #Access the Q-network from the model
for layer in q_network.parameters(): #Inspect the weights of the Q-networks
    print(layer.data) #Visualize weights or biases for each layer

replay_buffer = model.replay_buffer
obs = replay_buffer.observations
actions = replay_buffer.actions
rewards = replay_buffer.rewards

obs2 = torch.tensor(obs, dtype=torch.float32)
q_values = model.q_net(obs2)


# Read the log file
data = pd.read_csv(log_file)
print(data.head())

# Convert 'actions' from strings to lists of individual characters
data['actions'] = data['actions'].apply(lambda x: list(x))

# Create a figure
plt.figure(figsize=(18, 8))

# Plot rewards
plt.subplot(2, 2, 1)
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
plt.subplot(2, 2, 2)
plt.plot(data[' length'], label='Length per Episode', color='green')
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')
plt.grid(True)
plt.legend()

# Plot action frequency heatmap
plt.subplot(2, 2, 3)
all_actions = pd.Series([action for sublist in data['actions'] for action in sublist])
action_counts = all_actions.value_counts().sort_index()  # Sort by action for better visualization

# Create a bar plot of action counts
sns.barplot(x=action_counts.index, y=action_counts.values, palette='viridis')
plt.title('Action Frequency')
plt.xlabel('Action')
plt.ylabel('Count')

# Plot action sequences as a color-coded matrix
plt.subplot(2, 2, 4)
action_matrix = data['actions'].apply(lambda x: [ord(a) - ord('a') for a in x])  # Convert actions to numerical representation
max_length = action_matrix.apply(len).max()  # Get the max length of action sequences for padding
action_matrix = pd.DataFrame(action_matrix.tolist()).fillna(-1).astype(int)  # Pad with -1 for missing actions

# Display the action matrix as an image (each episode is a row, each action is a column)
plt.imshow(action_matrix, cmap='Set3', aspect='auto')
plt.colorbar(label='Action Code')  # Add colorbar to represent the action codes
plt.title('Action Sequences per Episode')
plt.xlabel('Action Step')
plt.ylabel('Episode')

# Example: plotting rewards from the replay buffer
plt.subplot(2,2,5)
plt.plot(rewards)
plt.title('Rewards from Replay Buffer')
plt.xlabel('Timestep')
plt.ylabel('Reward')

# Visualize the Q-values for each action
plt.subplot(2,2,6)
plt.plot(q_values.detach().numpy())
plt.title('Q-values for Different Actions')
plt.xlabel('Observation Index')
plt.ylabel('Q-value')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
