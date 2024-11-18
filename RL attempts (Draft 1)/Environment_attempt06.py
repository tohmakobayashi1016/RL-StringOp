import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import sys, os, json
from math import pi, cos, sin

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp
from compas_fd.solvers import fd_numpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Classes.feature_extraction import MeshFeature
from Classes.FormatConverter import FormatConverter
from Classes.PostProcessor import PostProcessor

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=100, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize classes
        self.format_converter  = FormatConverter()
        self.post_processor    = PostProcessor()
        
        #Initialize mesh and terminal state
        self.initial_mesh      = initial_mesh
        self.terminal_mesh     = self.load_terminal_mesh(terminal_mesh_json_path)
        self.terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices=False)

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
        self.observation_space = Dict({
            "vertices": self.node_space,
            "edge_index": self.edge_index_space,
            "edge_attr": self.edge_attr_space
        })

        #Initialize compas parameters
        self.lizard            = self.position_lizard(self.initial_mesh_copy)
        self.last_action       = None
        self.a_count           = 0
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
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.position_lizard(self.current_mesh)
        self.action_string = ['']
        self.a_count = 0
        self.copy_poles        = []
        self.current_poles     = []

        self.current_step = 0
        self.episode_number += 1
        print(f"Environment reset. Starting episode {self.episode_number}.")
        obs = self.get_state()
        info = {}
        return (obs, info)
    
    def step(self, action):
        self.current_step += 1
        action_letter = self.format_converter.from_discrete_to_letter([int(action)])
        self.action_string[0] += action_letter
        
        print(f"Step {self.current_step}: Action string - {self.action_string[0]}")

        #Position lizard
        self.lizard = self.position_lizard(self.initial_mesh_copy)

        #Debugging: Print vertices and faces before lizard_atp
        print(f"Before lizard_atp: Vertices: {list(self.current_mesh.vertices())}, Faces: {list(self.current_mesh.faces())}")
        
        #Apply the actions to initial_copy and store info in current_mesh
        tail, body, head = lizard_atp(self.initial_mesh_copy, self.lizard, self.action_string[0])
        self.copy_poles  = []

        for fkey in self.initial_mesh_copy.faces():
            fv = self.initial_mesh_copy.face_vertices(fkey)
            if len(fv) == 3:
                if body in fv:
                    self.copy_poles.append(self.initial_mesh_copy.vertex_coordinates(body))
                else:
                    'pbm identification pole'
                    self.copy_poles.append(self.initial_mesh_copy.vertex_coordinates(fv[0]))

        #Debugging: Print vertices and faces before post-process
        print(f"After lizard_atp: Vertices: {list(self.current_mesh.vertices())}, Faces: {list(self.current_mesh.faces())}")
        
        if not self.initial_mesh_copy.is_manifold():
            print('Mesh not manifold. Terminating episode.')
            reward = -0.01
            terminated = True
            truncated = True
            obs, info = self.reset()
            return obs, reward, terminated, truncated, info

        #Post-process the mesh 
        self.post_processor.postprocess(self.initial_mesh_copy)

        #Debugging: Print vertices and faces after post-process [INITIAL COPY]
        print(f"Initial copy after post-process: Vertices: {list(self.initial_mesh_copy.vertices())}, Faces: {list(self.initial_mesh_copy.faces())}")

        #Update self.current_mesh with self.initial_mesh_copy
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh_copy.to_vertices_and_faces())
        self.post_processor.postprocess(self.current_mesh)
        #self.current_mesh = self.initial_mesh_copy
        self.current_poles = self.copy_poles.copy
        
        #Update the list of vertices at the end of step
        self.update_vertices()

        #Debugging: Print vertices and faces after post-process [INITIAL COPY]
        print(f"Current mesh after post-process: Vertices: {list(self.current_mesh.vertices())}, Faces: {list(self.current_mesh.faces())}")

        terminated = self.is_terminal_state()
        truncated = self.current_step >= self.max_steps or len(self.action_string[0]) >= 50
        reward = self.calculate_reward(terminated)
        obs = self.get_state() #I think I also need to modify get_state
        
        if terminated:
            print("Episode terminated.")
            obs, info = self.reset()
            return obs, reward, terminated, truncated, info
        if truncated:
            print("Episode truncated.")
            obs, info = self.reset()
            return obs, reward, terminated, truncated, info


        #Reset initial_mesh_copy
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.copy_poles = []

        return obs, reward, terminated, truncated, {}
    
    def calculate_reward(self, done):
        #terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices = False)
        current_features = MeshFeature(self.current_mesh).categorize_vertices(display_vertices = False)

        #Extract degree information for boundary and interior vertices
        current_boundary_degree = current_features['boundary_vertices_by_degree']
        terminal_boundary_degree = self.terminal_features['boundary_vertices_by_degree']
        current_interior_degree = current_features['inside_vertices_by_degree']
        terminal_interior_degree = self.terminal_features['inside_vertices_by_degree']
        
        #Calculate the reward based on how close the current degrees are to the terminal degree
        reward = 0
        small_positive_reward = 0.01
        large_positive_reward = 1.0

        #Compare boundary vertices' degrees
        for degree, info in current_boundary_degree.items():
            if degree in terminal_boundary_degree:
                terminal_count = terminal_boundary_degree[degree]['count']
                current_count  = info['count']
                reward += small_positive_reward * min(terminal_count, current_count)

        #Compare inside vertices' degrees
        for degree, info in current_interior_degree.items():
            if degree in terminal_interior_degree:
                terminal_count = terminal_interior_degree[degree]['count']
                current_count  = info['count']
                reward += small_positive_reward * min(terminal_count, current_count)

        #Normalize the reward by the number of vertices 
        num_vertices = len(list(self.current_mesh.vertices()))
        reward /= num_vertices
        
        if done and self.is_terminal_state():
            print("Terminal state reached, adding large positive reward")
            reward += large_positive_reward

        return reward

    def is_terminal_state(self):
        #terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices=False)
        current_features  = MeshFeature(self.current_mesh).categorize_vertices(display_vertices=False)

        boundary_mismatches = []
        interior_mismatches = []

        for degree, info in self.terminal_features['boundary_vertices_by_degree'].items():
            if degree not in current_features['boundary_vertices_by_degree'] or \
            info ['count'] != current_features['boundary_vertices_by_degree'][degree]['count']:
                boundary_mismatches.append(degree)
            
        for degree, info in self.terminal_features['inside_vertices_by_degree'].items():
            if degree not in current_features['inside_vertices_by_degree'] or \
            info ['count'] != current_features['inside_vertices_by_degree'][degree]['count']:
                interior_mismatches.append(degree)
                
        if boundary_mismatches:
            print(f"Boundary degrees do not match: {boundary_mismatches}")
        if interior_mismatches:
            print(f"Interior degrees do not match: {interior_mismatches}")
        if not boundary_mismatches and not interior_mismatches:    
            print("Terminal state detected")
            return True
        return False
    
    def get_state(self):
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()], dtype=np.float32)
        
        if len(vertices) < self.max_vertices:
            padding = np.zeros((self.max_vertices - len(vertices), 3), dtype=np.float32)
            vertices = np.vstack((vertices, padding))
        vertices = vertices[:self.max_vertices]

        edges = []
        
        for u, v in self.current_mesh.edges():
            edges.append((u, v))
            edges.append((v, u))
            
        edge_index = np.array(edges, dtype=np.int32)
        edge_attr = np.ones((edge_index.shape[0], 2), dtype=np.float32)

        if edge_index.shape[0] < self.max_vertices * 4:
            padding_index = np.zeros((self.max_vertices * 4 - edge_index.shape[0], 2), dtype=np.int32)
            edge_index = np.vstack((edge_index, padding_index))
            padding_attr = np.zeros((self.max_vertices * 4 - edge_attr.shape[0], 2), dtype=np.float32)
            edge_attr = np.vstack((edge_attr, padding_attr))

        return {
            "vertices": vertices,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }
    def update_vertices(self):
        #Update the list of vertices based on the current mesh
        updated_vertices = {vkey: self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()}
        self.current_mesh.update_default_vertex_attributes(updated_vertices)