import numpy as np

def calculate_histogram_mse(current_histogram, terminal_histogram, degrees_to_consider):
    """
    Calculate the Mean Squared Error (MSE) between two histograms (current and terminal)
    for specified degrees.
    
    Parameters:
    - current_histogram: Dictionary representing the current mesh histogram
    - terminal_histogram: Dictionary representing the terminal mesh histogram
    - degrees_to_consider: List of keys (vertex degrees) to consider in the MSE calculation
    
    Returns:
    - mse: Mean Squared Error for the specified degrees
    """
    # Calculate squared errors for each degree to consider
    squared_errors = [
        (len(current_histogram[key]) - len(terminal_histogram[key])) ** 2
        for key in degrees_to_consider
    ]

    # Compute the mean of the squared errors
    mse = np.mean(squared_errors)
    return mse

# Example usage
if __name__ == "__main__":
    # Define example histograms for current and terminal mesh
    current_histogram = {
        'degree_2_vertices': [0, 2, 6, 8, 10],
        'degree_3_vertices': [1, 3, 5, 7],
        'degree_4_vertices': [4, 9],
        'degree_5_vertices': [],
        'degree_6_plus_vertices': []
    }

    terminal_histogram = {
        'degree_2_vertices': [2, 6, 8, 9],
        'degree_3_vertices': [1, 5, 7, 10, 11],
        'degree_4_vertices': [11],
        'degree_5_vertices': [14],
        'degree_6_plus_vertices': []
    }

    # Specify which degrees to consider in the MSE calculation
    degrees_to_consider = ['degree_2_vertices', 'degree_3_vertices', 'degree_4_vertices', 'degree_5_vertices', 'degree_6_plus_vertices']

    # Calculate the MSE between the two histograms
    mse_value = calculate_histogram_mse(current_histogram, terminal_histogram, degrees_to_consider)
    
    # Output the MSE result
    print(f"Mean Squared Error between current and terminal histograms: {mse_value}")
