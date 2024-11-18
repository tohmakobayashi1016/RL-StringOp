import os
import compas
from time import time
from math import pi, cos, sin
from collections import Counter


from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_quad.grammar.addition2 import add_strip_lizard
from compas_quad.grammar.lizard import string_generation_brute, string_generation_random, string_generation_structured, string_generation_evolution, Lizard
from compas_fd.solvers import fd_numpy
from compas_viewer.viewer import Viewer

# custom postprocessing function
def postprocessing(mesh):

    key2index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    index2key = {i: vkey for i, vkey in enumerate(mesh.vertices())}

    #print("Vertex Key to Index Mapping:")
    #for vkey, index in key2index.items():
        #print(f"Vertex Key: {vkey}, Index: {index}")
    #print("\nIndex to Vertex Key Mapping:")
    #for index, vkey in index2key.items():
        #print(f"Index: {index}, Vertex Key: {vkey}")

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

    #print("\nUpdated Vertex Coordinates:")
    #for vkey in mesh.vertices():
        #attr = mesh.vertex[vkey]
        #print(f"Vertex Key: {vkey}, Coordinates: ({attr['x']}, {attr['y']}, {attr['z']})")

    vertex_degrees = [mesh.vertex_degree(vkey) for vkey in mesh.vertices()]
    degree_distribution = Counter(vertex_degrees)
    print("Vertex Degree Distribution:", degree_distribution)
    face_valences = [len(mesh.face_vertices(fkey)) for fkey in mesh.faces()]
    valence_distribution = Counter(face_valences)
    print("Face Valence Distribution:", valence_distribution)
    




### parameters ###

input_mesh_refinement = 2  # densify the input 1-face quad mesh
output_mesh_refinement = 1  # densify the ouput quad mesh

# for 'given' production
add_given_strings = True
given_strings = ["atpttpttpttptta"]

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
        return [''.join(vector)]

# Create an instance of the class
converter = StringVectorConverter()

# Call the method with a string
string_to_vector = converter.from_string_to_vector(given_strings)

# Print the result
print(f"Input string: {given_strings}")
print(f"Output list: {string_to_vector}")



# for 'brute' force enumeration
add_brute_strings = False
brute_string_characters = '01'
brute_string_length = 8

# for 'random' generation
add_random_strings = False
random_string_characters = '01'
random_string_number = 5
random_string_length = 10
random_string_ratios = [0.5, 0.5]

# for 'structured' construction
add_structured_strings = False
structured_string_characters = 'atp'
structured_string_number = 100
structured_string_length = 8

# random evolution
add_evolution_strings = False
evolution_string_characters = '01'
evolution_string_number = 50
evolution_string_length = 20

postprocess = True
densify = True
array = True
view = True
export_json = True

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

#if view:
    #viewer = Viewer()
    #viewer.scene.add(mesh0)

### lizard - let's grooow! ###

# position lizard
for vkey in mesh0.vertices_on_boundary():
    if mesh0.vertex_degree(vkey) == 2:
        body = vkey
        tail, head = [nbr for nbr in mesh0.vertex_neighbors(vkey) if mesh0.is_vertex_on_boundary(nbr)]
    break
lizard = (tail, body, head)
print('lizard initial position', lizard)

# produce strings
strings = []
if add_given_strings:
    strings += string_to_vector
if add_brute_strings:
    strings += list(string_generation_brute(brute_string_characters, brute_string_length))
if add_random_strings:
    strings += list(string_generation_random(random_string_characters, random_string_number, random_string_length, ratios=random_string_ratios))
if add_structured_strings:
    strings += list(string_generation_structured(structured_string_characters, structured_string_number, structured_string_length))
if add_evolution_strings:
    strings += list(string_generation_evolution(evolution_string_characters, evolution_string_number, evolution_string_length))
print('{} strings: {}'.format(len(strings), strings))



# apply
t0 = time()
mesh2string = {}
for k, string in enumerate(strings):
    print(string)

    # modifiy topology
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(*mesh0.to_vertices_and_faces())
    tail, body, head = add_strip_lizard(mesh, lizard, string)
    
    poles = []
    for fkey in mesh.faces():
        fv = mesh.face_vertices(fkey)
        if len(fv) == 3:
            if body in fv:
                poles.append(mesh.vertex_coordinates(body))
            else:
                # warn if pole missing
                'pbm identification pole'
                poles.append(mesh.vertex_coordinates(fv[0]))
    
    # warn if mesh not manifold
    if not mesh.is_manifold():
        print('mesh not manifold')
        continue

    # export JSON
    if export_json:
        HERE = os.path.dirname(__file__)
        FILE = os.path.join(HERE, 'data/{}_{}.json'.format(input_mesh_refinement, string))
        mesh.to_json('C:/Users/footb/Desktop/Thesis/String-RL/Output/meaningful/{}.json'.format(string))

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

# results
t1 = time()
print('computation time {}s for {} meshes'.format(round(t1 - t0, 3), len(mesh2string)))

if array:
    n = len(mesh2string)
    for k, mesh in enumerate(mesh2string):
        n2 = int(n ** 0.5)
        i = int(k / n2)
        j = int(k % n2)
        mesh.move([1.5 * (i + 1), 1.5 * (j + 1), 0.0])

if view:
    viewer = Viewer()
    for mesh in mesh2string:
        viewer.scene.add(mesh)
    viewer.show()

print(mesh2string)