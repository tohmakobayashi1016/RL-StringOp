import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import sys, os

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Classes')))
from feature_extraction import MeshFeature

class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=100):
        super(MeshEnvironment, self).__init__()

        #Initialize mesh and terminal state
        self.initial_mesh = initial_mesh
        self.terminal_mesh = self.load_terminal_mesh_from_json(terminal_mesh_json_path)

        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())
        
        #Define action space
        self.actions = ['a', 't', 'p']
        self.action_string = ''
        self.action_space = spaces.Discrete(len(self.actions))

        #Define observation space 
        self.vertices_list = list(self.current_mesh.vertices())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.vertices_list)* 3,), dtype=np.float32)

        self.lizard = self.find_lizard(self.current_mesh)
        self.max_steps = max_steps
        self.current_step = 0   
        
        print("Environment initialized.")

    def load_terminal_mesh_from_json(self, json_path):
        print(f"Loading terminal mesh from {json_path}...")
        mesh = Mesh()
        mesh.from_json(json_path)
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
        self.action_string += self.actions[action]
        tail, body, head = self.lizard
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*self.initial_mesh.to_vertices_and_faces())

        lizard_atp(self.current_mesh, (tail, body, head), self.action_string)

        

        poles = []
        for fkey in self.current_mesh.faces():
            fv = self.current_mesh.face_vertices(fkey)
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
            return self.get_state().flatten().astype(np.float32), reward, terminated, truncated, {}


        self.current_step += 1

        reward = self.calculate_reward()
        terminated = self.is_terminal_state()
        truncated = self.current_step >= self.max_steps

        return self.get_state().flatten().astype(np.float32), reward, terminated, truncated, {}
    
    def calculate_reward(self):
        terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices()
        current_features = MeshFeature(self.current_mesh).categorize_vertices()

        #print(f"Terminal features: {terminal_features}")
        #print(f"Current features: {current_features}")
        
        reward = 0
        for key in terminal_features:
            term_boundary = terminal_features[key].get('boundary_vertices_by_degree', {})
            curr_boundary = current_features[key].get('boundary_vertices_by_degree', {})
            term_internal = terminal_features[key].get('inside_vertices_by_degree', {})
            curr_internal = current_features[key].get('inside_vertices_by_degree', {})

            # Calculate rewarrd based on similarity of vertex distributions
            for degree in term_boundary:
                reward -= abs(term_boundary.get(degree, {'count': 0})['count'] - curr_boundary.get(degree, {'count': 0})['count'])
            for degree in term_internal:
                reward -= abs(term_internal.get(degree, {'count': 0})['count'] - curr_internal.get(degree, {'count': 0})['count'])

        return reward
    
    def is_terminal_state(self):
        terminal_features = MeshFeature(self.terminal_mesh).categorize_vertices()
        current_features = MeshFeature(self.current_mesh).categorize_vertices()

        for key in terminal_features:
            term_boundary = terminal_features[key].get('boundary_vertices_by_degree', {})
            curr_boundary = current_features[key].get('boundary_vertices_by_degree', {})
            term_internal = terminal_features[key].get('inside_vertices_by_degree', {})
            curr_internal = current_features[key].get('inside_vertices_by_degree', {})

            #Check if all vertex distribution match
            for degree in term_boundary:
                if term_boundary.get(degree, {'count': 0})['count'] != curr_boundary.get(degree, {'count': 0})['count']:
                    return False
            for degree in term_internal:
                if term_internal.get(degree, {'count': 0})['count'] != curr_internal.get(degree, {'count': 0})['count']:
                    return False
                
        return True
    
    def get_state(self):
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()])
        return vertices