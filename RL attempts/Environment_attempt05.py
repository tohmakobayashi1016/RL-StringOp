import numpy as np 
import gymnasium as gym
from gymnasium.spaces import Graph, Box, Discrete
import sys, os, json
from math import pi, cos, sin

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp
from compas_fd.solvers import fd_numpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Classes')))
from feature_extraction import MeshFeature
from FormatConverter import FormatConverter
from PostProcessor import PostProcessor

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=100, max_vertices=50):
        super(MeshEnvironment, self).__init__()

        #Initialize mesh and terminal state
        self.initial_mesh      = initial_mesh
        self.terminal_mesh     = self.load_terminal_mesh_from_json(terminal_mesh_json_path)

        #Separate initial mesh for modification and current meh for tracking state
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())
        self.current_mesh      = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())

        #Define action space
        self.actions           = ['a', 't', 'p']
        self.action_string     = ''
        self.action_space      = Discrete(len(self.actions))

        #Define observation space 
        self.max_vertices      = max_vertices
        self.node_space        = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.edge_space        = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = Graph(node_space=self.node_space, edge_space=self.edge_space)

        self.lizard            = self.find_lizard(self.current_mesh)
        self.max_steps         = max_steps
        self.current_step      = 0   

        self.format_converter  = FormatConverter()
        self.post_processor    = PostProcessor()

        self.post_processor.postprocess(self.terminal_mesh)
        
        print("Environment initialized.")

    def load_terminal_mesh_from_json(self, json_path):
        print(f"Loading terminal mesh from {json_path}...")
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
        
        vertices_list = list(mesh.vertices())
        faces_list = list(mesh.faces())
        print(f"Terminal mesh vertices: {vertices_list}")
        print(f"Terminal mesh faces: {faces_list}")
        
        if not vertices_list or not faces_list:
            raise ValueError("Loaded mesh has no vertices or faces")
        
        return mesh
    
    def find_lizard(self, mesh):
        for vkey in mesh.vertices_on_boundary():
            if mesh.vertex_degree(vkey) == 2:
                body = vkey
                tail, head = [nbr for nbr in mesh.vertex_neighbors(vkey) if mesh.is_vertex_on_boundary(nbr)]
            break
        lizard = (tail, body, head)
        return lizard
    
    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.find_lizard(self.current_mesh)
        self.action_string = ''
        self.current_step = 0
        print("Environment reset.")
        obs = self.get_state()
        info = {}
        return (obs, info) if return_info else (obs, info)
    
    def step(self, action):
        self.current_step += 1
        
        action_letter = self.format_converter.from_discrete_to_letter([int(action)])
        self.action_string += action_letter

        print(f"Step {self.current_step}: Action string - {self.action_string}")

        #Apply the action to the current mesh IM CHECKING IF THIS WORKS ON INITIAL MESH
        self.initial_mesh_copy = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.find_lizard(self.initial_mesh_copy)
        tail, body, head = lizard_atp(self.initial_mesh_copy, self.lizard, self.action_string)

        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh_copy.to_vertices_and_faces())
        
        poles = []
        for fkey in self.initial_mesh_copy.faces():
            fv = self.initial_mesh_copy.face_vertices(fkey)
            if len(fv) == 3:
                if body in fv:
                    poles.append(self.current_mesh.vertex_coordinates(body))
                else:
                    'pbm identification pole'
                    poles.append(self.current_mesh.vertex_coordinates(fv[0]))
        
        if not self.current_mesh.is_manifold():
            print('Mesh not manifold. Terminating episode.')
            reward = -1.0
            terminated = True
            truncated = True
            return self.get_state(), reward, terminated, truncated, {}

        #Post-process the mesh 
        self.post_processor.postprocess(self.current_mesh)
        
        terminated = self.is_terminal_state()
        truncated = self.current_step >= self.max_steps or len(self.action_string) >= 20
        reward = self.calculate_reward(terminated)
        obs = self.get_state()

        return obs, reward, terminated, truncated, {}
    
    def calculate_reward(self, done):
        terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices = False)
        current_features = MeshFeature(self.current_mesh).categorize_vertices(display_vertices = False)

       # terminal_degrees = {vkey: self.terminal_mesh.vertex_degree(vkey) for vkey in self.terminal_mesh.vertices()}
       # current_degrees  = {vkey: self.current_mesh.vertex_degree(vkey) for vkey in self.current_mesh.vertices()}        
        
        #Extract degree information for boundary and interior vertices
        current_boundary_degree = current_features['boundary_vertices_by_degree']
        terminal_boundary_degree = terminal_features['boundary_vertices_by_degree']
        current_interior_degree = current_features['inside_vertices_by_degree']
        terminal_interior_degree = terminal_features['inside_vertices_by_degree']
        
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
        terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices=False)
        current_features  = MeshFeature(self.current_mesh).categorize_vertices(display_vertices=False)

        boundary_mismatches = []
        interior_mismatches = []

        for degree, info in terminal_features['boundary_vertices_by_degree'].items():
            if degree not in current_features['boundary_vertices_by_degree'] or \
            info ['count'] != current_features['boundary_vertices_by_degree'][degree]['count']:
                boundary_mismatches.append(degree)
            
        for degree, info in terminal_features['inside_vertices_by_degree'].items():
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
        self.post_processor.postprocess(self.current_mesh)
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()])
        
        if len(vertices) < self.max_vertices:
            padding = np.zeros((self.max_vertices - len(vertices), 3))
            vertices = np.vstack((vertices, padding))
        vertices = vertices[:self.max_vertices]

        edges = []
        edge_attrs = []

        for u, v in self.current_mesh.edges():
            edges.append((u, v))
            edges.append((v, u))
            edge_attrs.append([1.0])
            edge_attrs.append([1.0])
        
        edge_index = np.array(edges, dtype=np.int32)
        edge_attr  = np.array(edge_attrs, dtype=np.float32)

        if edge_index.shape[0] < self.max_vertices * 4:
            padding = np.zeros((self.max_vertices * 4 - edge_index.shape[0], 2), dtype=np.int32)
            edge_index = np.vstack((edge_index, padding))
            padding_attr = np.zeros((self.max_vertices * 4 - edge_attr.shape[0], 1), dtype=np.float32)
            edge_attr = np.vstack((edge_attr, padding_attr))
        
        return GraphInstance(nodes=vertices, edges=edge_attr, edge_links = edge_index)