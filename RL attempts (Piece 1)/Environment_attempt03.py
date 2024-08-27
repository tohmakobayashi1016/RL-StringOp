import numpy as np 
import gymnasium as gym
from gymnasium import spaces
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
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=100):
        super(MeshEnvironment, self).__init__()

        #Initialize mesh and terminal state
        self.initial_mesh      = initial_mesh
        self.terminal_mesh     = self.load_terminal_mesh_from_json(terminal_mesh_json_path)

        if not self.terminal_mesh:
            raise ValueError("Terminal mesh could not be loaded or is empty")
            
        self.current_mesh      = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())

        #Define action space
        self.actions           = ['a', 't', 'p']
        self.action_string     = ''
        self.action_space      = spaces.Discrete(len(self.actions))

        #Define observation space 
        self.vertices_list     = list(self.current_mesh.vertices())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.vertices_list)* 3,), dtype=np.float32)

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
                return (tail, body, head)
        return None
    
    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())
        self.lizard = self.find_lizard(self.current_mesh)
        self.action_string = ''
        self.current_step = 0
        print("Environment reset.")
        obs = self.get_state().flatten().astype(np.float32)
        return (obs, {}) if return_info else (obs, {})
    
    def step(self, action):
        self.current_step += 1
        
        action_letter = self.format_converter.from_discrete_to_letter([int(action)])
        self.action_string += action_letter

        print(f"Step {self.current_step}: Action chosen - {action_letter}")

        #Apply the action to the current mesh
        lizard_atp(self.current_mesh, self.lizard, self.action_string)
        
        poles = []
        for fkey in self.current_mesh.faces():
            fv = self.current_mesh.face_vertices(fkey)
            if len(fv) == 3:
                if self.lizard[1] in fv:
                    poles.append(self.current_mesh.vertex_coordinates(self.lizard[1]))
                else:
                    'pbm identification pole'
                    poles.append(self.current_mesh.vertex_coordinates(fv[0]))
        
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.current_mesh.to_vertices_and_faces())

        if not self.current_mesh.is_manifold():
            print('Mesh not manifold. Terminating episode.')
            reward = -1.0
            terminated = True
            truncated = True
            return self.get_state().flatten().astype(np.float32), reward, terminated, truncated, {}

        #Post-process the mesh 
        self.post_processor.postprocess(self.current_mesh)
        
        terminated = self.is_terminal_state()
        truncated = self.current_step >= self.max_steps
        reward = self.calculate_reward(terminated)

        return self.get_state().flatten().astype(np.float32), reward, terminated, truncated, {}
    
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
        #for key in terminal_features:
            #term_boundary = terminal_features[key].get('boundary_vertices_by_degree', {})
            #curr_boundary = current_features[key].get('boundary_vertices_by_degree', {})
            #term_internal = terminal_features[key].get('inside_vertices_by_degree', {})
            #curr_internal = current_features[key].get('inside_vertices_by_degree', {})

            # Calculate rewarrd based on similarity of vertex distributions
            #for degree in term_boundary:
                #reward -= abs(term_boundary.get(degree, {'count': 0})['count'] - curr_boundary.get(degree, {'count': 0})['count'])
            #for degree in term_internal:
                #reward -= abs(term_internal.get(degree, {'count': 0})['count'] - curr_internal.get(degree, {'count': 0})['count'])

        return reward

    
    def is_terminal_state(self):
        terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices(display_vertices=False)
        current_features  = MeshFeature(self.current_mesh).categorize_vertices(display_vertices=False)

        for key in terminal_features:
            term_boundary = terminal_features[key].get('boundary_vertices_by_degree', {})
            curr_boundary = current_features[key].get('boundary_vertices_by_degree', {})
            term_internal = terminal_features[key].get('inside_vertices_by_degree', {})
            curr_internal = current_features[key].get('inside_vertices_by_degree', {})

            #Check if all vertex distribution match
            for degree in term_boundary:
                if term_boundary.get(degree, {'count': 0})['count'] != curr_boundary.get(degree, {'count': 0})['count']:
                    print(f"Boundary degree {degree} does not match")
                    return False
            for degree in term_internal:
                if term_internal.get(degree, {'count': 0})['count'] != curr_internal.get(degree, {'count': 0})['count']:
                    print(f"Internal degree {degree} does not match")
                    return False
        print("Terminal state detected")
        return True
    
    def get_state(self):
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()])
        return vertices