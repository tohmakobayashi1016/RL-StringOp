import numpy as np
import random
from compas.datastructures import Mesh
from compas.geometry import centroid_polygon, area_polygon, length_vector, subtract_vectors
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_viewer.viewer import Viewer
from compas.colors import Color, ColorDict


class MeshFeature:
    def __init__(self, mesh, lizard=None):
        if isinstance(mesh, Mesh):
            self.mesh = mesh
            self.lizard = lizard
        else:
            raise ValueError("Input must be a compas Mesh object")

    def topological_features(self):
        features = {
            "number_of_vertices": self.mesh.number_of_vertices(),
            "number_of_edges": self.mesh.number_of_edges(),
            "number_of_faces": self.mesh.number_of_faces(),
            "vertex_degree": {key: self.mesh.vertex_degree(key) for key in self.mesh.vertices()},
            "face_areas": {fkey: area_polygon(self.mesh.face_coordinates(fkey)) for fkey in self.mesh.faces()},
            "face_centroids": {fkey: centroid_polygon(self.mesh.face_coordinates(fkey)) for fkey in self.mesh.faces()},
        }
        return features

    def isomorphic_features(self):
        return {
            "is_symmetric": self.check_symmetry(),
        }

    def check_symmetry(self):
        centroid = self.mesh.centroid()
        vertices = self.mesh.vertices_attributes('xyz')
        symmetric = all(subtract_vectors(centroid, v) == subtract_vectors(centroid, v) for v in vertices)
        return symmetric

    def extract_features(self):
        features = {
            "topological_features": self.topological_features(),
            "isomorphic_features": self.isomorphic_features()
        }
        return features

    def categorize_vertices(self):
        degree_histogram = [0] * 5 # List of length 5
        vertex_colors = ColorDict(default=Color.white())
        
        # Collect the degree of each vertex
        for vertex in self.mesh.vertices():
            degree = self.mesh.vertex_degree(vertex)

            #Fill histogram for degrees from 2 to 6 and above
            if degree== 2:
                degree_histogram[0] += 1
            elif degree== 3:
                degree_histogram[1] += 1
            elif degree== 4:
                degree_histogram[2] += 1
            elif degree== 5:
                degree_histogram[3] += 1
            else: #Degree 6 and above
                degree_histogram[4] += 1
                        
            #Assign a color based on the vertex degree
            vertex_colors[vertex] = self.get_color_for_degree(degree)

        # Print statement to display the categorized vertices
        print("Degree histogram:")
        for i, count in enumerate(degree_histogram):
            if i < 4:
                print(f"Degree {i+2}: {count} vertices")
            else:
                print(f"Degree 6 and above: {count} vertices")

        return {
            "degree_histogram": degree_histogram,
            "vertex_colors": vertex_colors
        }

    
    def get_color_for_degree(self, degree):
       #Assign colors to inside vertices based on the degree of boundary vertices
        if degree == 2:
            return Color.red()
        if degree == 3:
            return Color.cyan()
        elif degree == 4:
            return Color.magenta()
        elif degree == 5:
            return Color.yellow()
        else:
            return Color.orange()
       

    
