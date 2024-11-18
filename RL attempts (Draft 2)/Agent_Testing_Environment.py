from Mesh_Environment_simplified_string_mse_A_to_B_prateek import MeshEnvironment

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
   

# Path to initial mesh JSON file
initial_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\a.json'

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

# Initialize environment
env = MeshEnvironment(initial_mesh_json_path, terminal_mesh_json_path, max_steps = 4)

obs, info = env.reset()

action1 = 1

obs, reward, terminated, truncated, info = env.step(action1)

action2 = 1

obs, reward, terminated, truncated, info = env.step(action2)

action3 = 0

obs, reward, terminated, truncated, info = env.step(action3)

print(info)


