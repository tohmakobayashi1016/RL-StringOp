import numpy as np 
import gymnasium as gym
from gymnasium import spaces

from compas.datastructures import Mesh
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import lizard_atp


class MeshEnvironment(gym.Env):
    def __init__(self, initial_mesh, terminal_mesh_json_path, max_steps=100):
        super(MeshEnvironment, self).__init__()
        self.initial_mesh = initial_mesh
        self.terminal_mesh = self.load_terminal_mesh_from_json(terminal_mesh_json_path)
        self.current_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*initial_mesh.to_vertices_and_faces())
        self.actions = ['a', 't', 'p']
        self.lizard = self.find_lizard(self.current_mesh)
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Discrete(len(self.actions))
        self.vertices_list = list(self.current_mesh.vertices())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.vertices_list)* 3,), dtype=np.float32)
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
        self.current_step = 0
        print("Environment reset.")
        obs = self.get_state().flatten().astype(np.float32)
        return obs, {}  
    
    def step(self, action):
        action_str = self.actions[action]
        tail, body, head = self.lizard
        lizard_atp(self.current_mesh, (tail, body, head), action_str)

        self.current_step += 1

        reward = self.calculate_reward()
        terminated = self.is_terminal_state()
        truncated = self.current_step >= self.max_steps

        return self.get_state().flatten().astype(np.float32), reward, terminated, truncated, {}
    
    def calculate_reward(self):
        return 1.0 if self.is_terminal_state() else -0.1
    
    def is_terminal_state(self):
        return (self.current_mesh.to_vertices_and_faces() == self.terminal_mesh.to_vertices_and_faces())
    
    def get_state(self):
        vertices = np.array([self.current_mesh.vertex_coordinates(vkey) for vkey in self.current_mesh.vertices()])
        return vertices