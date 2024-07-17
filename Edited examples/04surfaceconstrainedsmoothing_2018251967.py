from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from compas.datastructures import Mesh

import compas_rhino
from compas_rhino.conversions import MeshArtist

from compas_rhino.geometry import RhinoPoint
from compas_skeleton.rhino.surface import RhinoSurface
from compas_skeleton.rhino.curve import RhinoCurve

from compas_skeleton.rhino.constraints import automated_smoothing_surface_constraints
from compas_skeleton.rhino.constraints import automated_smoothing_constraints
from compas_skeleton.rhino.smoothing import constrained_smoothing

# ==============================================================================
# Input
# ==============================================================================

# Get input data.
mesh_guid = compas_rhino.select_mesh("Select a mesh to smooth.")
srf_guid = compas_rhino.select_surface("Select a surface as smoothing constraint.")
crv_guids = compas_rhino.select_curves("Select curves as smoothing constraints.")
pt_guids = compas_rhino.select_points("Select points as smoothing constraints.")

# Wrap the inputs.
mesh = Mesh.from_vertices_and_faces(compas_rhino.rs.MeshVertices(mesh_guid), compas_rhino.rs.MeshFaceVertices(mesh_guid))
surface = RhinoSurface.from_guid(srf_guid)
curves = [RhinoCurve.from_guid(guid) for guid in crv_guids]
points = [RhinoPoint.from_guid(guid) for guid in pt_guids]

# ==============================================================================
# Postprocess the result
# ==============================================================================

# Constrain mesh components to the feature geometry.
constraints = automated_smoothing_surface_constraints(mesh, surface)
constraints.update(
    automated_smoothing_constraints(mesh, rhinopoints=points, rhinocurves=curves))

# Get the number of iterations for smoothing the quad mesh.
k = compas_rhino.rs.GetInteger("Define the number of iterations for smoothing the pattern.", 10)

# Smooth with constraints.
constrained_smoothing(
    mesh, kmax=k, damping=0.5, constraints=constraints, algorithm="area")

# ==============================================================================
# Visualization
# ==============================================================================

artist = MeshArtist(mesh)#, layer="Singular::SmoothMesh")
#artist.clear_layer()
artist.draw_mesh()
