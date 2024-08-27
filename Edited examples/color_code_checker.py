import os, sys, json, random
from math import pi, cos, sin

from compas.datastructures import Mesh
from compas_quad.grammar.addition2 import lizard_atp
from compas_fd.solvers import fd_numpy
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_viewer.viewer import Viewer
#from compas_view2.app import App

from compas.colors import Color
from compas.geometry import Scale, Translation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Classes.feature_extraction import MeshFeature

def postprocessing(mesh):

    key2index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    index2key = {i: vkey for i, vkey in enumerate(mesh.vertices())}

    # map mesh boundary vertices to circle
    fixed = mesh.vertices_on_boundary()[:-1]
    n = len(fixed)
    for i, vkey in enumerate(fixed):
        attr = mesh.vertex[vkey]
        attr['x'] = 0.5 * cos(i / n * 2 * pi)
        attr['y'] = 0.5 * sin(i / n * 2 * pi)
        attr['z'] = 0

    # form finding with force density method
    vertices = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
    edges = [(key2index[u], key2index[v]) for u, v in mesh.edges()]
    fixed = [key2index[vkey] for vkey in fixed]
    q = [1.0] * len(edges)
    loads = [[0.0, 0.0, 1.0 / len(vertices)]] * len(vertices)
    result = fd_numpy(vertices=vertices, edges=edges, fixed=fixed, forcedensities=q, loads=loads)
    xyz = result.vertices
    for i, (x, y, z) in enumerate(xyz):
        vkey = index2key[i]
        attr = mesh.vertex[vkey]
        attr['x'] = x
        attr['y'] = y
        attr['z'] = z
        
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp\terminal_mesh\trial.json'

viewer = Viewer()
mesh = Mesh.from_json(terminal_mesh_json_path)
print(mesh)

mesh1 = CoarsePseudoQuadMesh.from_vertices_and_faces(*mesh.to_vertices_and_faces())
postprocessing(mesh1)

feature_extractor = MeshFeature(mesh1)
results = feature_extractor.categorize_vertices()

vertex_colors = results["vertex_colors"]
vertex_colors_dict = {key: color for key, color in vertex_colors.items()}
#vertexcolor = {vkey: Color.blue() for vkey in mesh1.vertices()}
facecolor = {fkey: Color.green() for fkey in mesh1.faces()}
linecolor = {ekey: Color.black() for ekey in mesh1.edges()}



viewer.scene.add(
    mesh1, 
    show_points=True, 
    use_vertexcolors=True, 
    facecolor=facecolor, 
    linecolor=linecolor, 
    pointcolor=vertex_colors_dict,
    #pointcolor=vertexcolor,
    pointsize=1.0
    )

#mesh2 = Mesh.from_vertices_and_faces(*mesh1.to_vertices_and_faces())



#mesh2.to_json('C:/Users/footb/Desktop/Thesis/String-RL/Output/color.json')

#mesh2.move([3, 3, 0.0])
#viewer.scene.add(mesh2)


viewer.show()