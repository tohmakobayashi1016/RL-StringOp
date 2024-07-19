import json

def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Load and print the JSON data to verify its structure
json_path = "C:/Users/footb/Desktop/Thesis/String-RL/Output/RL-attempt-01/trial.json"
data = load_json_data(json_path)
print(json.dumps(data, indent=4))

