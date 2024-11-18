from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN

from Mesh_Environment_simplified import MeshEnvironment
from compas_quad.datastructures import CoarsePseudoQuadMesh

# Load the trained model
model = DQN.load("DQN_simplified_model")

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

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

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
