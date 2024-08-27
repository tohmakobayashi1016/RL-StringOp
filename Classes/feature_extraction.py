import numpy as np
import random
from compas.datastructures import Mesh
from compas.geometry import centroid_polygon, area_polygon, length_vector, subtract_vectors
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas_viewer.viewer import Viewer
from compas.colors import Color, ColorDict


class MeshFeature:
    def __init__(self, mesh):
        if isinstance(mesh, Mesh):
            self.mesh = mesh
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

    def categorize_vertices(self, display_vertices=True):
        boundary_vertices = set(self.mesh.vertices_on_boundary())
        inside_vertices = set(self.mesh.vertices()) - boundary_vertices
        
        boundary_vertices_by_degree = {}
        inside_vertices_by_degree = {}

        vertex_colors = ColorDict(default=Color.white())

        boundary_color = Color.red()
        for vertex in boundary_vertices:
            vertex_colors[vertex] = boundary_color

        # Categorize boundary vertices by degree
        for vertex in boundary_vertices:
            degree = self.mesh.vertex_degree(vertex)
            if degree not in boundary_vertices_by_degree:
                boundary_vertices_by_degree[degree] = {"count": 0, "vertices": set()}
            boundary_vertices_by_degree[degree]["count"] += 1
            boundary_vertices_by_degree[degree]["vertices"].add(vertex)

        # Categorize inside vertices by degree and assign colors
        for vertex in inside_vertices:
            degree = self.mesh.vertex_degree(vertex)
            if degree not in inside_vertices_by_degree:
                inside_vertices_by_degree[degree] = {"count": 0, "vertices": set()}
            inside_vertices_by_degree[degree]["count"] += 1
            inside_vertices_by_degree[degree]["vertices"].add(vertex)
            vertex_colors[vertex] = self.get_color_for_inside_degree(degree)

        result = {
            "boundary_vertices": boundary_vertices,
            "boundary_vertices_by_degree": boundary_vertices_by_degree,
            "inside_vertices_by_degree": inside_vertices_by_degree,
            "vertex_colors": vertex_colors
        }

        # Print statement to display the categorized vertices
        if display_vertices:
            print("Boundary vertices by degree:")
            for degree, info in boundary_vertices_by_degree.items():
                print(f"Degree {degree}: Count = {info['count']}, Vertices = {info['vertices']}")

            print("Interior vertices by degree:")
            for degree, info in inside_vertices_by_degree.items():
                print(f"Degree {degree}: Count = {info['count']}, Vertices = {info['vertices']}")
        else:
            print("Boundary vertices by degree:")
            for degree, info in boundary_vertices_by_degree.items():
                print(f"Degree {degree}: Count = {info['count']}")

            print("Interior vertices by degree:")
            for degree, info in inside_vertices_by_degree.items():
                print(f"Degree {degree}: Count = {info['count']}")

        return result

    
    def get_color_for_inside_degree(self, degree):
       #Assign colors to inside vertices based on the degree of boundary vertices
        if degree == 3:
            return Color.cyan()
        elif degree == 4:
            return Color.magenta()
        elif degree == 5:
            return Color.yellow()
        else:
            return Color.orange()
       

    
