from math import pi, cos, sin
from compas_fd.solvers import fd_numpy

class PostProcessor:
    def __init__(self):
        pass

    def postprocess(self, mesh):
        key2index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
        index2key = {i: vkey for i, vkey in enumerate(mesh.vertices())}

        if not key2index or not index2key:
            print("Error: Mesh contains no vertices.")
            return
        
        # Map mesh boundary vertices to circle
        fixed = mesh.vertices_on_boundary()[:-1]
        n = len(fixed)
        for i, vkey in enumerate(fixed):
            attr = mesh.vertex[vkey]
            attr['x'] = 0.5 * cos(i / n * 2 * pi)
            attr['y'] = 0.5 * sin(i / n * 2 * pi)
            attr['z'] = 0

        # Form finding with force density method
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