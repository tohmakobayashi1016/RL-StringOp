import compas_quad
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import add_strip_lizard
from compas_quad.grammar.lizard import Lizard
from compas_viewer.viewer import Viewer
from math import pi, cos, sin
from compas_fd.solvers import fd_numpy
import os
import numpy as np
from time import time

#Want to make a controller that can implement grammar commands via marker (not implemented)
#1. intialize environment
#2. establish actions
#3. process and implement single step

# custom postprocessing function for visualization
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

    

input_mesh_refinement = 1  # densify the input 1-face quad mesh
output_mesh_refinement = 1  # densify the ouput quad mesh

vertices = [[0,0,0], [0,1,0], [1,1,0], [1,0,0]]
faces = [[0, 1, 2, 3]]
coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

# denser mesh, does this do anything here?
coarse.collect_strips()
coarse.strips_density(input_mesh_refinement)
coarse.densification()
mesh0 = coarse.dense_mesh()
mesh0.collect_strips()

# position lizard
for vkey in mesh0.vertices_on_boundary():
    if mesh0.vertex_degree(vkey) == 2:
        body = vkey
        tail, head = [nbr for nbr in mesh0.vertex_neighbors(vkey) if mesh0.is_vertex_on_boundary(nbr)]
    break
lizard = (tail, body, head)
print('lizard initial position', lizard)


t0 = time()
mesh2string = {}
view = True
export_json = True
postprocess = True
densify = True

# for 'given' production
add_given_strings = True
given_strings = ["a", "at", "ata"]

class StringVectorConverter:
    def from_string_to_vector(self, string):
        vector = []
        for k in string:
            if k == 't':
                vector.append("00")
            elif k == 'p':
                vector.append("01")
            elif k == 'a':
                vector.append("10")
            elif k == 'd':
                vector.append("11")
        # Join the elements of the vector into a single string and return it as a list with one item
        return ''.join(vector)

# Create an instance of the class
converter = StringVectorConverter()

# Call the method with a string
strings = []
for s in given_strings:
    strings.append(converter.from_string_to_vector(s))

print(strings)

    
for k, string in enumerate(strings):
    print(string)
    #How can I implement a print(lizard position) to keep track of where the marker is? HOW DO I KEEP TRACK OF MARKER POSITION?
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*mesh0.to_vertices_and_faces()) #Isn't this redundant to break the mesh into vertices and faces, to reconstruct? or is this part of the update
    tail, body, head = add_strip_lizard(mesh, lizard, string)
    poles = []

    for fkey in mesh.faces():   #what does this do?
        fv = mesh.face_vertices(fkey)
        if len(fv) == 3:
            if body in fv:
                poles.append(mesh.vertex_coordinates(body))
            else:
                #warn if pole missing
                'pbm identificaion pole'
                poles.append(mesh.vertex_coordinates(fv[0]))
    
    if not mesh.is_manifold():
        print('mesh not manifold')
        continue
    # export JSON
    if export_json:
        HERE = os.path.dirname(__file__)
        FILE = os.path.join(HERE, 'data/{}_{}.json'.format(input_mesh_refinement, string))
        mesh.to_json('C:/Users/footb/Desktop/Thesis/String-RL/Output/06/{}.json'.format(string))

    # geometry and density processing
    if postprocess:
        postprocessing(mesh)
        if densify:
            mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(*mesh.to_vertices_and_faces(), poles=poles)
            mesh.collect_strips()
            mesh.strips_density(output_mesh_refinement)
            mesh.densification()
            mesh = mesh.dense_mesh()
            postprocessing(mesh)

    mesh2string[mesh] = string

if view:
    viewer = Viewer()
    for mesh in mesh2string:
        viewer.scene.add(mesh)
    viewer.show()

print(mesh2string)