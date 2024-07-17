import os, compas
from time import time
from math import pi, cos, sin
from collections import Counter, namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import add_strip_lizard
from compas_quad.grammar.lizard import string_generation_brute, string_generation_random, string_generation_structured, string_generation_evolution
from compas_fd.solvers import fd_numpy
from compas_viewer.viewer import Viewer

### parameters ###

input_mesh_refinement = 2
view = True

### intialise ###

# dummy mesh with a single quad face
vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

# denser mesh
coarse.collect_strips()
coarse.strips_density(input_mesh_refinement)
coarse.densification()
mesh0 = coarse.dense_mesh()
mesh0.collect_strips()

if view:
    viewer = Viewer()
    viewer.scene.add(mesh0)

viewer.show()