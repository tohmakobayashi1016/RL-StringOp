import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import sys, os, json

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Classes.feature_extraction_histogram_granular import MeshFeature
from Classes.FormatConverter import FormatConverter
from Classes.PostProcessor import PostProcessor
from Classes.distance_functions import DistanceFunctions

"""

Custom quad mesh environment for integrating compas_quad and RL algorithms.

Key variables of simplified version: topological features and reward function tuning

Topological features: Histogram of singularity degrees and mesh distance

String features: Levenshtein distance

Reward function: Mean square error (MSE), ...

"""

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=5, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize meshes
        self.initial_mesh = initial_mesh
        self.terminal_mesh = self.load_terminal_mesh(terminal_mesh_json_path)
        self.terminal_histogram = MeshFeature(self.terminal_mesh).categorize_vertices()

        #Create a working copy of initial mesh for modification (Do I need this?)
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())

        #Action space and observation space
        self.action_space = Discrete(3)
        self.action_string = ['']
        self.max_vertices = max_vertices
        self.create_observation_space()

        #Initialize classes and compas parameters
        self.format_converter = FormatConverter()
        self.distance_calc = DistanceFunctions()
        self.lizard = self.position_lizard(self.current_mesh)
        self.max_steps = max_steps
        self.current_step = 0

        print("Environment initialized.")

    def create_observation_space(self):
        #Topological features
        self.vertex_degree_histogram_space = Box(low=0, high=self.max_vertices, shape=(5,), dtype=np.int32)

        #Distance features
        self.levenshtein_distance = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.mesh_distance = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = Dict({
            "degree_histogram": self.vertex_degree_histogram_space,
            "levenshtein_distance": self.levenshtein_distance,
            "mesh_distance": self.mesh_distance,

        })

    def load_terminal_mesh(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data['dtype'] != 'compas.datastructures/Mesh':
            raise ValueError("Incorrect JSON format")
        
        mesh = Mesh()
        vertices = data['data']['vertex']
        faces = data['data']['face']

        for key, value in vertices.items():
            mesh.add_vertex(int(key), x=value['x'], y=value['y'], z=value['z'])

        for key, value in faces.items():
            mesh.add_face(value)

        return mesh
    
    def position_lizard(self, mesh):
        for vkey in mesh.vertices_on_boundary():
            if mesh.vertex_degree(vkey) == 2:
                body = vkey
                tail, head = [nbr for nbr in mesh.vertex_neighbors(vkey) if mesh.is_vertex_on_boundary(nbr)]
            break
        lizard = (tail, body, head) 
        return lizard

    def reset(self, seed=None, return_info=False, options=None): #Check the implementation for seed
        super().reset(seed=seed)
        # Reset the mesh state and environment variables
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.position_lizard(self.current_mesh)
        self.action_string = ['']
        self.current_step = 0

        obs, _ = self.get_state()
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        penalty = 0

        # Copy of the initial mesh to apply the current action
        initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())

        try:
            # Convert action and apply to mesh
            action_letter = self.format_converter.from_discrete_to_letter([int(action)])
            self.action_string[0] += action_letter
            print(f"Step {self.current_step}: Action string - {self.action_string[0]}")
            tail, body, head = lizard_atp(initial_mesh_copy, self.lizard, self.action_string[0])
            
            if body is None or head is None:
                print(f"Error: Invalid lizard state detected.")
                reward = -100 # Penalize invalid state
                terminated = True
                truncated = False
                return self.get_state()[0], reward, terminated, truncated, {}
            
            # Update the current msh with the result of the action
            self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh_copy.to_vertices_and_faces())

            # Check if the episode is finished
            terminated = self.is_terminal_state()
            truncated = self.current_step >= self.max_steps

            # Calculate the reward
            reward = self.calculate_reward(terminated)

            return self.get_state()[0], reward, terminated, truncated, {}
        
        except ValueError as e:
            reward = -500
            terminated = True
            truncated = False
            return self.get_state()[0], reward, terminated, truncated, {"error": str(e)}
        
    def calculate_reward(self, done):
        # Calculate vertex degree histogram for the current mesh
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()

        # Only consider degrees 3 and higher for MSE (TRIAL)
        degrees_to_consider = ['degree_3_vertices', 'degree_4_vertices', 'degree_5_vertices', 'degree_6_plus_vertices']

        # Compute mean squared rror (MSE) between terminal and current mesh
        mse = sum(np.mean((len(current_histogram[key]) - len(self.terminal_histogram[key])) **2)
                  for key in degrees_to_consider)
        
        # Distance-based rewards (Levenshtein + mesh distance)
        obs, _ = self.get_state()
        distance_reward = -1 * (obs['levenshtein_distance'][0] + obs['mesh_distance'][0]) **2

        # Time-step penalty
        time_step_penalty = -5.0

        # Define the terminal string
        terminal_string = 'atpta' # Examplle target string

        # Matching action reward
        action_string = ''.join(self.action_string)
        matching_reward = 0
        for i, action_char in enumerate(action_string):
            if i < len(terminal_string) and action_char == terminal_string[i]:
                matching_reward += 50

        # Total reward: negative MSE + time-step penalty + distance reward
        reward = -mse + time_step_penalty + distance_reward + matching_reward

        # Additional reward for reaching terminal state
        if done:
            print("Terminal state reached, adding large positive reward")
            reward += 205
        return reward
    
    def is_terminal_state(self):
        # Get the current degree histogram to compare with terminal mesh layou
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()

        # Compare histograms directly
        match_degree_2 = len(current_histogram['degree_2_vertices']) == len(self.terminal_histogram['degree_2_vertices'])
        match_degree_3 = len(current_histogram['degree_3_vertices']) == len(self.terminal_histogram['degree_3_vertices'])
        match_degree_4 = len(current_histogram['degree_4_vertices']) == len(self.terminal_histogram['degree_4_vertices'])
        match_degree_5 = len(current_histogram['degree_5_vertices']) == len(self.terminal_histogram['degree_5_vertices'])
        match_degree_6_plus = len(current_histogram['degree_6_plus_vertices']) == len(self.terminal_histogram['degree_6_plus_vertices'])

        done = match_degree_2 and match_degree_3 and match_degree_4 and match_degree_5 and match_degree_6_plus
        return done
            
    def get_state(self):
        # Compute the singularity degree histogram
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()
        degree_histogram = np.array([
            len(current_histogram['degree_2_vertices']),
            len(current_histogram['degree_3_vertices']),
            len(current_histogram['degree_4_vertices']),
            len(current_histogram['degree_5_vertices']),
            len(current_histogram['degree_6_plus_vertices'])
        ], dtype=np.int32)

        # Calculate Levenshtein and mesh distances
        current_string = [''.join(self.action_string)]
        #print(f" Action String: {current_string}")
        terminal_string = ['atpta'] # Target string
        levenshtein_distance = self.distance_calc.levenshtein_distance(current_string, terminal_string)[0]
        #print(f"Levenshtein distance: {levenshtein_distance}")
        mesh_distance = self.distance_calc.mesh_distance(current_string, terminal_string)[0]
        #print(f"Mesh distance: {mesh_distance}")

        obs = {
            "degree_histogram": degree_histogram,
            "levenshtein_distance": np.array([levenshtein_distance], dtype=np.float32),
            "mesh_distance": np.array([mesh_distance], dtype=np.float32),
        }

        return obs, True

