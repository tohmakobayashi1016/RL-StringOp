from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

from Mesh_Environment_simplified import MeshEnvironment
from compas_quad.datastructures import CoarsePseudoQuadMesh
import time

# Load the trained model
model = DQN.load("DQN_simplified_model")

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atpta.json'

#Initialize
input_mesh_refinement = 2  # densify the input 1-face quad mesh

# dummy mesh with a single quad face
vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

# denser mesh
coarse.collect_strips()
coarse.strips_density(input_mesh_refinement)
coarse.densification()
initial_mesh = coarse.dense_mesh()
initial_mesh.collect_strips()

# Load your environment with the same setup used for training
env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps=5)

# Reset the environment to get the initial observation
obs, _ = env.reset()

# Variable to track total reward in this episode
total_reward = 0

# Run one full episode
done = False
while not done:
    # Use the model to predict the action
    action, _states = model.predict(obs)
    
    # Take the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Accumulate the reward
    total_reward += reward
    
    # Optionally render the environment (if applicable)
    # env.render()  # Uncomment if the environment supports rendering
    
    # Print the step results
    print(f"Action taken: {action}, Reward received: {reward}, Total reward: {total_reward}")
    
    # Check if the episode is done
    done = terminated or truncated

print(f"Episode finished. Total reward: {total_reward}")
