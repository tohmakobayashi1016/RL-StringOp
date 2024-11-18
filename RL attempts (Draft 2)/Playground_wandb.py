import torch
import torch.nn.functional as F

def verify_cross_entropy_loss():
    # Simulated current degree histogram (observed from the environment)
    current_histogram = torch.tensor([4, 5, 1, 1, 0], dtype=torch.float32)  # Replace with actual observation
    print(f"Current Histogram: {current_histogram}")

    # Normalize current histogram to get probabilities (cross-entropy expects probabilities)
    current_histogram_prob = current_histogram / current_histogram.sum()
    print(f"Normalized Current Histogram (as probabilities): {current_histogram_prob}")

    # Simulated terminal degree histogram (target or ground truth)
    terminal_histogram = torch.tensor([4, 5, 1, 1, 0], dtype=torch.float32)  # Replace with actual terminal histogram
    print(f"Terminal Histogram: {terminal_histogram}")

    # Convert terminal histogram to a class label (argmax), simulating a classification target
    terminal_class = terminal_histogram.unsqueeze(0).argmax(dim=1)  # Get the index of the maximum value as class
    print(f"Terminal Class (index of max value): {terminal_class}")

    # Calculate cross-entropy loss
    cross_entropy_loss = F.cross_entropy(current_histogram_prob.unsqueeze(0), terminal_class)
    print(f"Cross-Entropy Loss: {cross_entropy_loss.item()}")

if __name__ == "__main__":
    verify_cross_entropy_loss()
