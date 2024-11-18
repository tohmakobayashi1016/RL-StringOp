import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import sys, os, json, torch
import torch.nn.functional as F

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

Hybrid reward function: String + Histogram

"""

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize meshes
        self.initial_mesh = initial_mesh
        #self.initial_mesh = self.load_mesh(initial_mesh_json_path)
        self.initial_histogram = MeshFeature(self.initial_mesh).categorize_vertices()
        self.terminal_mesh = self.load_mesh(terminal_mesh_json_path)
        self.terminal_histogram = MeshFeature(self.terminal_mesh).categorize_vertices()

        #Create a working copy of initial mesh for modification (Do I need this?)
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())

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

        #Limit for lizard positions
        lizard_position_high = float('inf')

        lizard_position = Box(low=0, high=lizard_position_high, shape=(3,), dtype=np.int32)

        self.observation_space = Dict({
            'degree_histogram': self.vertex_degree_histogram_space,
            'lizard_position': lizard_position
        })

    def load_mesh(self, json_path):
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

        self.lizard_position = {
            'tail': self.lizard[0],
            'body': self.lizard[1],
            'head': self.lizard[2]
        }

        self.action_string = ['']
        self.current_step = 0

        print("Environment is resetting")
        obs, _ = self.get_state()
        return obs, {}
    
    def step(self, action):
        # Copy of the initial mesh to apply the current action
        initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())

        terminated, truncated = False, False

        try:
            # Convert action and apply to mesh
            action_letter = self.format_converter.from_discrete_to_letter([int(action)])
            self.action_string[0] += action_letter
            print(f"Step {self.current_step}: Action string - {self.action_string[0]}")
            tail, body, head = lizard_atp(initial_mesh_copy, self.lizard, self.action_string[0])
            
            if body is None or head is None:
                raise TypeError
            
            self.lizard_position = {
                'tail': tail,
                'body': body,
                'head': head
            }

            # Update the current msh with the result of the action
            self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh_copy.to_vertices_and_faces())

            # Incremental timer
            self.current_step += 1

            # Check if the episode is finished
            terminated = self.is_terminal_state()
            truncated = self.current_step >= self.max_steps
        
        except (ValueError, TypeError):
            print(f"Error detected in compas_quad. Truncating the episode.")
            truncated = True
        
        # Calculate the reward
        reward = self.calculate_reward(terminated, truncated)
        
        # Log the observation before returning and before any potential reset
        obs = self.get_state()[0]
        print(f"Final observation: {obs}")
        print(f"Termination condition: {terminated}; Truncation condition: {truncated}.")

        # Log the final observation into info dict
        info = {}

        if self.current_step == self.max_steps or terminated == True:
            final_degree_histogram = obs.get("degree_histogram")
            final_lizard_position = obs.get("lizard_position")
            info["degree_histogram"] = final_degree_histogram
            info["lizard_position"] = final_lizard_position
            print("Final observations recorded.")

        return obs, reward, terminated, truncated, info
        
    def calculate_reward(self, done, truncated):
        # Calculate vertex degree histogram for the current mesh
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()

        # Only consider degrees 3 and higher for MSE (TRIAL)
        degrees_to_consider = ['degree_2_vertices','degree_3_vertices', 'degree_4_vertices', 'degree_5_vertices', 'degree_6_plus_vertices']

        # Compute mean squared rror (MSE) between terminal and current mesh
        squared_errors = [
            (len(current_histogram[key]) - len(self.terminal_histogram[key])) ** 2
            for key in degrees_to_consider
        ]
        
        mse = np.mean(squared_errors)

        # Calculate string and mesh distances
        current_string = [''.join(self.action_string)]
        terminal_string = ['atpttpta'] # Target string
        string_distance = self.distance_calc.levenshtein_distance(current_string, terminal_string)[0]

        # Time-step penalty
        time_step_penalty = -1 

        # Add penalty if the final design step is unchanged
        histogram_change = 0.0

        if done or truncated:
            # Compare current histogram with initial histogram
            identical_to_initial = all(
                len(current_histogram[key]) == len(self.initial_histogram[key])
                for key in degrees_to_consider
            )
            
            if identical_to_initial:
                print("Final design step is unchanged from initial histogram, adding penatly.")
                histogram_change = -2
            else:
                histogram_change = -time_step_penalty + (30 -string_distance - mse)
                print(f"Final design step is changed, applying reward: {histogram_change}")

        # Total reward: negative MSE + time-step penalty + distance reward
        reward = - mse - string_distance + time_step_penalty + histogram_change

        # Additional reward for reaching terminal state
        if done:
            print("Terminal state reached, adding large positive reward")
            reward += 100        
        return reward
    
    def is_terminal_state(self):
        # If checking action sequence in the forward direction
        current_string = [''.join(self.action_string)]
        terminal_string = ['atpttpta'] # Target string
        match_string = current_string == terminal_string
        
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

        # Alternative initial position to handle KeyErrors / ValueError/ TypeError
        initial_tail = 1
        initial_body = 0
        initial_head = 3

        # Include lizard's positions
        current_position = self.lizard_position
        tail = current_position.get('tail') if current_position.get('tail') is not None else initial_tail
        body = current_position.get('body') if current_position.get('body') is not None else initial_body
        head = current_position.get('head') if current_position.get('head') is not None else initial_head

        lizard_position = np.array([tail, body, head], dtype=np.int32)

        obs = {
            "degree_histogram": degree_histogram,
            "lizard_position": lizard_position
            }

        return obs, True

