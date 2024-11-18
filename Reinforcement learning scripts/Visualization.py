import csv

def read_column_to_list(csv_file_path, column_name, prepend_letter='a'):
    # Initialize an empty list to store the items
    items_list = []

    # Open the CSV file and read its content
    with open(csv_file_path, mode='r') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csvfile)
        
        # Iterate over each row in the CSV
        for row in csv_reader:
            value = row[column_name].strip()  # Strip to remove any surrounding spaces
            if value:  # Only process non-empty strings
                value = prepend_letter + value  # Prepend the letter
                items_list.append(value)

    return items_list

# Example usage
csv_file = r"C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\observation_PPO_mesh_three_step_mse.csv"  # Windows file path with raw string literal
column = 'action_letters'   # Replace with the column name you want to read
items = read_column_to_list(csv_file, column, prepend_letter='a')
print(items)