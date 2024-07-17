from compas.datastructures import Mesh
import compas_quad
from compas_quad.datastructures import CoarsePseudoQuadMesh
import os
import numpy as np
import pygame
from gymnasium import spaces
import gymnasium as gym

class QuadMeshEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size        # size of square grid
        self.window_size = 512  # size of PyGame window

        self.observation_space = spaces.Dict(
            {
                "agent" : spaces.Graph(node_space=spaces.Box(0, size - 1, shape=(2,)), edge_space=(2)),
                "target" : spaces.Graph(node_space=spaces.Box(0, size - 1, shape=(2,)), edge_space=(2)),
            }
        )

        #4 actions, "pivot" "turn" "add" "delete"
        self.action_space = spaces.Discrete(4)

        self.action_to_string = {
            0:
            1:
            2:
            3:
        }