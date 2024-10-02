import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Mesh_Environment_simplified import MeshEnvironment
from compas_quad.datastructures import CoarsePseudoQuadMesh

# Initialize a test environment
def setup_environment():
    # Set up a dummy initial mesh with a single quad face
    vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
    faces = [[0, 1, 2, 3]]
    coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

    # densify the mesh for the initial input
    input_mesh_refinement = 2
    coarse.collect_strips()
    coarse.strips_density(input_mesh_refinement)
    coarse.densification()
    initial_mesh = coarse.dense_mesh()

    # Set the path to your terminal mesh (replace with your actual path)
    terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

    # Initialize the environment
    env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps=5)

    return env

# Test function to step through the environment and verify rewards
def test_reward_function(env, action_sequence):
    env.reset()  # Reset environment to the initial state
    total_reward = 0
    action_string = ''

    # Step through the environment using the provided action sequence
    for i, action in enumerate(action_sequence):
        obs, reward, done, truncated, info = env.step(action)
        action_string += env.format_converter.from_discrete_to_letter([int(action)])

        print(f"Step {i+1}: Action = {env.format_converter.from_discrete_to_letter([int(action)])}")
        print(f"Reward: {reward}, Total Reward: {total_reward + reward}")
        print(f"Current Action String: {action_string}")
        print(f"Levenshtein Distance: {obs['levenshtein_distance'][0]}")
        print(f"Mesh Distance: {obs['mesh_distance'][0]}")
        print(f"Degree Histogram: {obs['degree_histogram']}")
        print("-" * 50)

        total_reward += reward
        if done or truncated:
            print(f"Episode finished after {i+1} steps.")
            break

    print(f"Final Total Reward: {total_reward}")
    return total_reward

# Test the reward function using a sample action sequence
if __name__ == "__main__":
    # Set up the environment
    env = setup_environment()

    # Example action sequence (replace with actual action numbers)
    # Assuming 0, 1, and 2 correspond to 'a', 't', 'p' in the action space
    action_sequence = [0, 1, 1, 0, 2]  # Modify this based on your action space and goal string

    # Run the test
    test_reward_function(env, action_sequence)
