import pandas as pd

# Path to the log file
log_file = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\training_log_x.csv'

# Read the CSV file
data = pd.read_csv(log_file)

# Extract the 'actions' column as a list of strings
actions_list = data['actions'].tolist()

# Display the formatted list
print(actions_list)


