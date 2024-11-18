import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import sys, os, json
from math import pi, cos, sin, sqrt
from compas_viewer.viewer import Viewer

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp
from compas_fd.solvers import fd_numpy
from compas.colors import Color

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Classes.feature_extraction_histogram_granular import MeshFeature
from Classes.FormatConverter import FormatConverter
from Classes.PostProcessor import PostProcessor
from Classes.distance_functions import DistanceFunctions

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=5, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize classes
        self.format_converter  = FormatConverter()
        self.post_processor    = PostProcessor()
        self.distance_calc     = DistanceFunctions()
        
        #Initialize mesh and terminal state
        self.initial_mesh      = initial_mesh
        self.terminal_mesh     = self.load_terminal_mesh(terminal_mesh_json_path)
        self.terminal_histogram = MeshFeature(self.terminal_mesh).categorize_vertices()

        #Separate initial mesh for modification and current mesh for tracking state
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())
        self.current_mesh      = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())

        #Define action space
        self.action_space      = Discrete(3)
        self.action_string     = [''] 
        
        # Define observation space using Dict
        self.max_vertices      = max_vertices
        self.create_observation_space()

        #Initialize compas parameters
        self.lizard            = self.position_lizard(self.initial_mesh_copy)
        self.last_action       = None
        self.a_count           = 0
        self.p_count           = 0
        self.t_count           = 0
        #self.d_count           = 0
        self.copy_poles        = []
        self.current_poles     = []

        #Initialize model parameters
        self.max_steps         = max_steps
        self.current_step      = 0   
        self.episode_number    = 0
   
        print("Environment initialized.")
    
    def create_observation_space(self):
        #Topological features
        self.vertex_degree_histogram_space = Box(low=0, high=self.max_vertices, shape=(5,), dtype=np.int32)
        #Geometrical features
        self.node_space = Box(low=-np.inf, high=np.inf, shape=(self.max_vertices, 3), dtype=np.float32)
        self.edge_index_space = Box(low=0, high=self.max_vertices, shape=(self.max_vertices * 4, 2), dtype=np.int32)
        self.edge_attr_space = Box(low=0, high=1, shape=(self.max_vertices * 4, 2), dtype=np.float32)
        self.face_space = Box(low=0, high=self.max_vertices, shape=(self.max_vertices*2, 4), dtype=np.int32)
        #distance features
        self.levenshtein_distance = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        self.mesh_distance = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.observation_space = Dict({
            "vertices": self.node_space,
            "edge_index": self.edge_index_space,
            "edge_attr": self.edge_attr_space,
            "faces": self.face_space,
            "degree_histogram": self.vertex_degree_histogram_space,
            "levenshtein_distance": self.levenshtein_distance,
            "mesh_distance": self.mesh_distance,
        })

    def load_terminal_mesh(self, json_path):
        print(f"Loading terminal mesh from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        if data['dtype'] != 'compas.datastructures/Mesh':
            raise ValueError("Incorrect JSON format")
        
        mesh = Mesh()
        #self.post_processor.postprocess(mesh)
        vertices = data['data']['vertex']
        faces = data['data']['face']

        for key, value in vertices.items():
            mesh.add_vertex(int(key), x=value['x'], y=value['y'], z=value['z'])
        
        for key, value in faces.items():
            mesh.add_face(value)
        
        vertices_list = list(mesh.vertices())
        faces_list = list(mesh.faces())
        print(f"Terminal mesh vertices: {vertices_list}")
        print(f"Terminal mesh faces: {faces_list}")
        
        if not vertices_list or not faces_list:
            raise ValueError("Loaded mesh has no vertices or faces")
        
        return mesh
    
    def position_lizard(self, mesh):
        for vkey in mesh.vertices_on_boundary():
            if mesh.vertex_degree(vkey) == 2:
                body = vkey
                tail, head = [nbr for nbr in mesh.vertex_neighbors(vkey) if mesh.is_vertex_on_boundary(nbr)]
            break
        lizard = (tail, body, head)
        print('Lizard initial position', lizard) 
        return lizard
    
    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        
        #reset environment
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.position_lizard(self.current_mesh)
        self.action_string = ['']
        self.a_count = 0
        self.p_count = 0
        self.t_count = 0
        #self.d_count = 0
        
        self.copy_poles        = []
        self.current_poles     = []

        self.current_step = 0
        self.episode_number += 1
        print(f"Environment reset. Starting episode {self.episode_number}.")
        obs, valid = self.get_state()

        if not valid:
            print("Invalid state detected during reset.")
            obs = None
            
        info = {}
        return (obs, info)
    
    def step(self, action):
        penalty = 0

        try:
            #Step increment and action execution
            self.current_step += 1
            action_letter = self.format_converter.from_discrete_to_letter([int(action)])
            self.action_string[0] += action_letter
        
            print(f"Step {self.current_step}: Action string - {self.action_string[0]}")

            #Track consecutive 'a' selection
            if action_letter == 'a':
                self.a_count += 1
                self.p_count = 0
                self.t_count = 0
                self.d_count = 0
            elif action_letter == 'p':
                self.a_count = 0
                self.p_count += 1
                self.t_count = 0
                self.d_count = 0
            elif action_letter == 't':
                self.a_count = 0
                self.p_count = 0
                self.t_count += 1
                self.d_count = 0
                
            #else:
                #self.d_count = 1 #Reset counter when different action is applied
                
            
            #Penalize consecutive 'a' selection (9-18) NOT NEEDED FOR SIMPLIFIED MODEL?
            penalty = self.penalize_repeated_actions()
            
            #Position lizard
            self.lizard = self.position_lizard(self.initial_mesh_copy)

            #Execute action and modify the mesh: apply actions to initial_copy, store info. in current_mesh
            tail, body, head = lizard_atp(self.initial_mesh_copy, self.lizard, self.action_string[0])

            if body is None or head is None:
                print(f"Error: Invalid lizard state detected: tail={tail}, body={body}, head={head}")
                #Handle the error
                reward = -500
                terminated = True
                truncated = False
                obs, _ = self.get_state()
                info = {"error": "Lizard state invalid due to None value."}
                return obs, reward + penalty, terminated, truncated, info
            
            #Post-process and update the mesh after action
            self.process_mesh_after_action(body)

            #if not valid:
            #    return self.terminate_episode(obs, -0.5 + penalty, "Face with more than 4 vertices detected.")
           
            #if not self.validate_mesh_faces(self.initial_mesh_copy):
            #    return self.terminate_episode(obs, -0.5 + penalty, "Invalid faces detected.")
            
            #if not self.initial_mesh_copy.is_manifold():
            #    return self.terminate_episode(obs, -0.01 + penalty, "Mesh not manifold.")

            #Calculate the reward and check if the episode is terminated or truncated 
            terminated = self.is_terminal_state()
            truncated = self.current_step >= self.max_steps or len(self.action_string[0]) >= 5
            reward = self.calculate_reward(terminated) + penalty

            if terminated or truncated:
                obs, info = self.reset()
                return obs, reward, terminated, truncated, info

        except ValueError as e:
            print(f"Error: {e}")
            reward = -500
            terminated = True
            truncated = False
            obs, _ = self.get_state()
            info = {"error": str(e)}
            return obs, reward + penalty, terminated, truncated, info

        #Get the current observation state and return results
        obs, _ = self.get_state()
        info = {}
        #Debugging
        print(f"Levenshtein Distance after action: {obs['levenshtein_distance'][0]}")
        #levenshtein_distance = obs['levenshein_distance'][0]
        print(f"Mesh Distance after action: {obs['mesh_distance'][0]}")
        #mesh_distance = obs['mesh_distance'][0]

        #Reset initial_mesh_copy for next step
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.copy_poles = []

        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self, done):
        # Current vertex degree histograms
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()

        # Separated MSE
        mse_degree_2 = np.mean((len(current_histogram['degree_2_vertices']) - len(self.terminal_histogram['degree_2_vertices']))**2)
        mse_degree_3 = np.mean((len(current_histogram['degree_3_vertices']) - len(self.terminal_histogram['degree_3_vertices']))**2)
        mse_degree_4 = np.mean((len(current_histogram['degree_4_vertices']) - len(self.terminal_histogram['degree_4_vertices']))**2)
        mse_degree_5 = np.mean((len(current_histogram['degree_5_vertices']) - len(self.terminal_histogram['degree_5_vertices']))**2)
        mse_degree_6_plus = np.mean((len(current_histogram['degree_6_plus_vertices']) - len(self.terminal_histogram['degree_6_plus_vertices']))**2)

        # Ensue both histograms contain numerical values by converting to integers
        #terminal_histogram = [int(t) for t in terminal_histogram]
        #current_histogram = [int(c) for c in current_histogram]
        
        # Calculate least squares error between terminal and current histograms
        current_mse = mse_degree_2 + mse_degree_3 + mse_degree_4 + mse_degree_5 + mse_degree_6_plus
        
        #Just seeing if this works
        obs, valid = self.get_state()
        levenshtein_distance = obs['levenshtein_distance'][0]
        mesh_distance = obs['mesh_distance'][0]

        distance_reward = -1*(levenshtein_distance + mesh_distance)**2

        # end of distance checking

        # Time-step penalty
        time_step_penalty = -5.0

        # Total reward: negative LSE + time-step penalty
        reward = -current_mse + time_step_penalty + distance_reward #distance reward trial
        
        if done and self.is_terminal_state():
            print("Terminal state reached, adding large positive reward")
            reward += 200

        return reward

    def process_mesh_after_action(self, body):
        """
        Post-process the mesh after the action and update the current state.
        """
        for fkey in self.initial_mesh_copy.faces():
            fv = self.initial_mesh_copy.face_vertices(fkey)
            if len(fv) == 3:
                self.copy_poles.append(self.initial_mesh_copy.vertex_coordinates(body if body in fv else fv[0]))

        #Post process the mesh
        #self.post_processor.postprocess(self.initial_mesh_copy)

        #Update the current mesh
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh_copy.to_vertices_and_faces())
        #self.post_processor.postprocess(self.current_mesh)

        #Update the current poles and vertices
        self.current_poles = self.copy_poles.copy()
        self.update_vertices()

    def terminate_episode(self, obs, reward, error_message):

        """
        Handle episode termination with an error message and return values.
        """
        print(f"{error_message} Terminating episode.")
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {"error": error_message}

    def penalize_repeated_actions(self):

        """
        
        Penalize repeated selection of repeated action
        i.e. if 'a' is selected consectuively, apply punishment.

        """
        max_consecutive_act = 3
        penalty = 0

        if self.a_count >= max_consecutive_act:
            print(f"Penalty applied for {self.a_count} consecutive 'a' actions.")
            penalty -= 0.05 * self.a_count

        if self.p_count >= max_consecutive_act:
            print(f"Penalty applied for {self.p_count} consecutive 'p' actions.")
            penalty -= 0.1 * self.p_count
        
        if self.t_count >= max_consecutive_act:
            print(f"Penalty applied for {self.t_count} consecutive 't' actions.")
            penalty -= 0.1 * self.t_count

        if self.d_count >= max_consecutive_act:
            print(f"Penalty applied for {self.d_count} consecutive 'd' actions.")
            penalty -= 0.1 * self.d_count

        return penalty


    def is_terminal_state(self):
        # Terminal and current vertex degree histograms
        current_histogram  = MeshFeature(self.current_mesh).categorize_vertices()

        # Compare histograms directly
        match_degree_2 = len(current_histogram['degree_2_vertices']) == len(self.terminal_histogram['degree_2_vertices'])
        match_degree_3 = len(current_histogram['degree_3_vertices']) == len(self.terminal_histogram['degree_3_vertices'])
        match_degree_4 = len(current_histogram['degree_4_vertices']) == len(self.terminal_histogram['degree_4_vertices'])
        match_degree_5 = len(current_histogram['degree_5_vertices']) == len(self.terminal_histogram['degree_5_vertices'])
        match_degree_6_plus = len(current_histogram['degree_6_plus_vertices']) == len(self.terminal_histogram['degree_6_plus_vertices'])

        return match_degree_2 and match_degree_3 and match_degree_4 and match_degree_5 and match_degree_6_plus
    
    def get_state(self):
        #Information extraction
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()], dtype=np.float32)
        edges = np.array(list(self.current_mesh.edges()), dtype=np.int32)
        edge_attr = np.ones((edges.shape[0],2), dtype=np.float32)

        faces = []
        for fkey in self.current_mesh.faces():
            face_vertices = list(self.current_mesh.face_vertices(fkey))
            if len(face_vertices) == 4:
                #Valid faces
                faces.append(face_vertices)
            elif len(face_vertices) < 4:
                #Pad faces with fewer than 4 vertices
                padded_face = face_vertices + [-1] * (4 - len(face_vertices))
                faces.append(padded_face)
            else: 
                #If face has more than 4 vertices
                return None, False
        
        faces = np.array(faces, dtype=np.int32)
        
        #Handle padding
        if vertices.shape[0] < self.max_vertices:
            padding = np.zeros((self.max_vertices - vertices.shape[0], 3), dtype=np.float32)
            vertices = np.vstack((vertices, padding))

        if edges.shape[0] < self.max_vertices*4:
            padding_edges = np.zeros((self.max_vertices*4 - edges.shape[0], 2), dtype=np.int32)
            edges = np.vstack((edges, padding_edges))
            padding_attr = np.zeros((self.max_vertices*4 - edge_attr.shape[0], 2), dtype=np.float32)
            edge_attr = np.vstack((edge_attr, padding_attr))

        if faces.shape[0] < self.max_vertices*2:
            padding_faces = np.zeros((self.max_vertices*2 - faces.shape[0], 4), dtype=np.int32)
            faces = np.vstack((faces, padding_faces))    

        # Compute the singulaity degree histogram for the current mesh
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()
        degree_histogram = np.array([
            len(current_histogram['degree_2_vertices']),
            len(current_histogram['degree_3_vertices']),
            len(current_histogram['degree_4_vertices']),
            len(current_histogram['degree_5_vertices']),
            len(current_histogram['degree_6_plus_vertices'])
        ], dtype=np.int32)

        # Convert the current mesh state and terminal state into strings for distance calculations
        current_string=[''.join(self.action_string)]
        terminal_string = ['atta'] #True string, given

        # Calculate Levenshtein and mesh distances
        levenshtein_distance = self.distance_calc.levenshtein_distance(current_string, terminal_string)[0]
        mesh_distance = self.distance_calc.mesh_distance(current_string, terminal_string)[0]

        #Debugging
        print(f"Action String: {current_string}")
        print(f"Terminal String: {terminal_string}")
        print(f"Levenshtein Distance: {levenshtein_distance}")
        print(f"Mesh distance: {mesh_distance}")

        obs = {
            "vertices": vertices,
            "edge_index": edges,
            "edge_attr": edge_attr,
            "faces": faces,
            "degree_histogram": degree_histogram,
            "levenshtein_distance": np.array([levenshtein_distance], dtype=np.float32),
            "mesh_distance": np.array([mesh_distance], dtype=np.float32),
        }

        return obs, True #Return a valid obs and a success flag
    

    def update_vertices(self):
        #Update the list of vertices based on the current mesh
        updated_vertices = {vkey: self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()}
        self.current_mesh.update_default_vertex_attributes(updated_vertices)
    
    def calculate_reward_component(self, component):
        """
        Return the requested reward component.
        """
        # Current vertex degree histograms
        current_histogram = MeshFeature(self.current_mesh).categorize_vertices()
        mse_degree_2 = np.mean((len(current_histogram['degree_2_vertices']) - len(self.terminal_histogram['degree_2_vertices']))**2)
        mse_degree_3 = np.mean((len(current_histogram['degree_3_vertices']) - len(self.terminal_histogram['degree_3_vertices']))**2)
        mse_degree_4 = np.mean((len(current_histogram['degree_4_vertices']) - len(self.terminal_histogram['degree_4_vertices']))**2)
        mse_degree_5 = np.mean((len(current_histogram['degree_5_vertices']) - len(self.terminal_histogram['degree_5_vertices']))**2)
        mse_degree_6_plus = np.mean((len(current_histogram['degree_6_plus_vertices']) - len(self.terminal_histogram['degree_6_plus_vertices']))**2)
        current_mse = mse_degree_2 + mse_degree_3 + mse_degree_4 + mse_degree_5 + mse_degree_6_plus

        # Distance rewards
        obs, valid = self.get_state()
        levenshtein_distance = obs['levenshtein_distance'][0]
        mesh_distance = obs['mesh_distance'][0]
        distance_reward = -1 * (levenshtein_distance + mesh_distance)**2

        # Time-step penalty
        time_step_penalty = -5.0

        if component == "mse":
            return current_mse
        elif component == "distance_reward":
            return distance_reward
        elif component == "time_step_penalty":
            return time_step_penalty
        else:
            raise ValueError(f"Unknown reward component: {component}")
