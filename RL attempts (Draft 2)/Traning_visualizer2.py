import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Path to the log file
log_file = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\training_log_exp_ver.csv'


# Read the log file
data = pd.read_csv(log_file)
print("Log file path:", log_file)
print(data.head())
plt.clf

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

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

