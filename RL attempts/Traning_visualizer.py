import matplotlib.pyplot as plt
import pandas as pd

# Path to the log file
log_file = 'training_log.csv'

# Read the log file
data = pd.read_csv(log_file)
print(data.head())

# Plot the learning curves
plt.figure(figsize=(12, 6))

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(data['reward'])
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

# Plot lengths
plt.subplot(1, 2, 2)
plt.plot(data['length'])
plt.title('Episode Lengths')
plt.xlabel('Episode')
plt.ylabel('Length')

plt.tight_layout()
plt.show()
