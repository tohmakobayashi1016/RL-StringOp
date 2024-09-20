import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import sys, os, json
from math import pi, cos, sin
from compas_viewer.viewer import Viewer

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp
from compas_fd.solvers import fd_numpy
from compas.colors import Color

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Classes.feature_extraction_histogram_version import MeshFeature
from Classes.FormatConverter import FormatConverter
from Classes.PostProcessor import PostProcessor

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=5, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize classes
        self.format_converter  = FormatConverter()
        self.post_processor    = PostProcessor()
        
        #Initialize mesh and terminal state
        self.initial_mesh      = initial_mesh
        self.terminal_mesh     = self.load_terminal_mesh(terminal_mesh_json_path)
        self.terminal_histogram, _ = MeshFeature(self.terminal_mesh).categorize_vertices()

        #Separate initial mesh for modification and current mesh for tracking state
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())
        self.current_mesh      = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())

        #Define action space
        self.action_space      = Discrete(3)
        self.action_string     = [''] 
        
        # Define observation space using Dict
        self.max_vertices = max_vertices
        self.node_space = Box(low=-np.inf, high=np.inf, shape=(max_vertices, 3), dtype=np.float32)
        self.edge_index_space = Box(low=0, high=max_vertices, shape=(max_vertices * 4, 2), dtype=np.int32)
        self.edge_attr_space = Box(low=0, high=1, shape=(max_vertices * 4, 2), dtype=np.float32)
        self.face_space = Box(low=0, high=max_vertices, shape=(max_vertices*2, 4), dtype=np.int32)
        self.observation_space = Dict({
            "vertices": self.node_space,
            "edge_index": self.edge_index_space,
            "edge_attr": self.edge_attr_space,
            "faces": self.face_space
        })

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

        #self.previous_lse = 100

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
            #if action_letter == 'a':
                #self.a_count += 1
                #self.p_count = 0
                #self.t_count = 0
                #self.d_count = 0
            #elif action_letter == 'p':
                #self.a_count = 0
                #self.p_count += 1
                #self.t_count = 0
                #self.d_count = 0
            #elif action_letter == 't':
                #self.a_count = 0
                #self.p_count = 0
                #self.t_count += 1
                #self.d_count = 0
                
            #else:
                #self.d_count = 1 #Reset counter when different action is applied
                
            
            #Penalize consecutive 'a' selection (9-18) NOT NEEDED FOR SIMPLIFIED MODEL
            #penalty = self.penalize_repeated_actions()
            
            #Position lizard
            self.lizard = self.position_lizard(self.initial_mesh_copy)

            #Execute action and modify the mesh: apply actions to initial_copy, store info. in current_mesh
            tail, body, head = lizard_atp(self.initial_mesh_copy, self.lizard, self.action_string[0])

            if body is None or head is None:
                print(f"Error: Invalid lizard state detected: tail={tail}, body={body}, head={head}")
                #Handle the error
                reward = -0.5
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
            reward = -0.5
            terminated = True
            truncated = False
            obs, _ = self.get_state()
            info = {"error": str(e)}
            return obs, reward + penalty, terminated, truncated, info

        #Get the current observation state and return results
        obs, _ = self.get_state()
        info = {}

        #Reset initial_mesh_copy for next step
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.copy_poles = []

        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self, done):
        # Terminal and current vertex degree histograms
        terminal_result = MeshFeature(self.terminal_mesh).categorize_vertices()
        current_result = MeshFeature(self.current_mesh).categorize_vertices()

        terminal_histogram = terminal_result['degree_histogram']
        current_histogram = current_result['degree_histogram']

        # Ensue both histograms contain numerical values by converting to integers
        terminal_histogram = [int(t) for t in terminal_histogram]
        current_histogram = [int(c) for c in current_histogram]
        
        # Calculate least squares error between terminal and current histograms
        current_lse = sum((t-c)**2 for t,c in zip(terminal_histogram, current_histogram))

        # Get the previouse LSE, if available
        #previous_lse = getattr(self, 'previous_lse', 100)

        # Time-step penalty
        time_step_penalty = -2.0

        # Total reward: negative LSE + time-step penalty
        reward = -current_lse + time_step_penalty
        
        if done and self.is_terminal_state():
            print("Terminal state reached, adding large positive reward")
            reward += 100

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
        terminal_result = MeshFeature(self.terminal_mesh).categorize_vertices()
        current_result  = MeshFeature(self.current_mesh).categorize_vertices()

        terminal_histogram = terminal_result['degree_histogram']
        current_histogram = current_result['degree_histogram']

        # Compare histograms directly
        if terminal_histogram == current_histogram:
            print("Terminal state detected")
            return True
        else:
            print("Vertex degree histograms do not match:")
            print(f"Terminal: {terminal_histogram}")
            print(f"Current: {current_histogram}")
            return False
    
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

        obs = {
            "vertices": vertices,
            "edge_index": edges,
            "edge_attr": edge_attr,
            "faces": faces
        }

        return obs, True #Return a valid obs and a success flag
    

    def update_vertices(self):
        #Update the list of vertices based on the current mesh
        updated_vertices = {vkey: self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()}
        self.current_mesh.update_default_vertex_attributes(updated_vertices)