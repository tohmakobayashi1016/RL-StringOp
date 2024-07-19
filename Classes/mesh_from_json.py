from compas.datastructures import Mesh
from compas_viewer import Viewer
import os
import sys
sys.path.append(os.path.abspath('Classes'))
from feature_extraction import MeshFeature
from compas_quad.datastructures import CoarsePseudoQuadMesh

file_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\RL-attempt-01\attatatattatta.trial.json'

mesh0 = Mesh.from_json(file_path)

mesh_features = MeshFeature(mesh0)
# Categorize vertices with display_vertices=True
#categorized_vertices_with_display = mesh_features.categorize_vertices(display_vertices=True)

# Categorize vertices with display_vertices=False
categorized_vertices_without_display = mesh_features.categorize_vertices(display_vertices=False)



initial_mesh_vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
initial_mesh_faces = [[0, 1, 2, 3]]
initial_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(initial_mesh_vertices, initial_mesh_faces)

mesh_features2 = MeshFeature(initial_mesh)

# Categorize vertices with display_vertices=True
#categorized_vertices_with_display = mesh_features2.categorize_vertices(display_vertices=True)

# Categorize vertices with display_vertices=False
#categorized_vertices_without_display = mesh_features2.categorize_vertices(display_vertices=False)

viewer = Viewer()
viewer.scene.add(mesh0)
viewer.show()


