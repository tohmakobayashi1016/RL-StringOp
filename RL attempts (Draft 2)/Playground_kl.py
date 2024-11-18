import torch
import torch.nn.functional as F
import numpy as np

# Define the KL-Divergence reward function
def calculate_kl_divergence_reward(current_histogram, terminal_histogram):
    # Only consider degrees 2 and higher for KL-Divergence
    degrees_to_consider = ['degree_2_vertices', 'degree_3_vertices', 'degree_4_vertices', 'degree_5_vertices', 'degree_6_plus_vertices']

    # Extract the raw counts from the current and terminal histograms
    current_counts = torch.tensor([current_histogram[key] for key in degrees_to_consider], dtype=torch.float32)
    terminal_counts = torch.tensor([terminal_histogram[key] for key in degrees_to_consider], dtype=torch.float32)

    # Normalize the histograms to get probability distributions (add small epsilon to avoid zero probabilities)
    epsilon = 1e-5
    current_distribution = (current_counts + epsilon) / (current_counts.sum() + epsilon)
    terminal_distribution = (terminal_counts + epsilon) / (terminal_counts.sum() + epsilon)

    # Apply KL-Divergence (Note: F.kl_div expects log-probabilities for the first argument)
    kl_divergence = F.kl_div(current_distribution.log(), terminal_distribution, reduction='batchmean')

    # Time-step penalty
    time_step_penalty = -5.0

    # Placeholder for design change reward/penalty
    change_reward_or_penalty = 25.0  # Assume a reward for design change

    # Total reward: negative KL-divergence + time-step penalty + reward/penalty for changes
    reward = -kl_divergence.item() + time_step_penalty + change_reward_or_penalty

    return reward

# Example histograms
initial_histogram = {
    'degree_2_vertices': 4,
    'degree_3_vertices': 4,
    'degree_4_vertices': 1,
    'degree_5_vertices': 0,
    'degree_6_plus_vertices': 0
}

# Current histogram example (replace with test data)
current_histogram = {
    'degree_2_vertices': 4,
    'degree_3_vertices': 4,
    'degree_4_vertices': 1,
    'degree_5_vertices': 0,
    'degree_6_plus_vertices': 0
}

# Terminal histogram example (target state)
terminal_histogram = {
    'degree_2_vertices': 4,
    'degree_3_vertices': 6,
    'degree_4_vertices': 3,
    'degree_5_vertices': 0,
    'degree_6_plus_vertices': 0
}

# Calculate KL-divergence reward
reward = calculate_kl_divergence_reward(current_histogram, terminal_histogram)
print(f"KL-Divergence Reward: {reward}")

# Test case when the current histogram is identical to the terminal histogram
identical_histogram = {
    'degree_2_vertices': 4,
    'degree_3_vertices': 6,
    'degree_4_vertices': 3,
    'degree_5_vertices': 0,
    'degree_6_plus_vertices': 0
}

# Calculate KL-divergence reward for identical histograms
reward_identical = calculate_kl_divergence_reward(identical_histogram, terminal_histogram)
print(f"KL-Divergence Reward (Identical Histograms): {reward_identical}")
