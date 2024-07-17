from compas.datastructures import Mesh
from compas_viewer import Viewer

file_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\RL-attempt-01\trial.json'

mesh0 = Mesh.from_json(file_path)

viewer = Viewer()
viewer.scene.add(mesh0)
viewer.show()

