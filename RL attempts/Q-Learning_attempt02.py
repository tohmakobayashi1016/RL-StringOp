from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from compas_quad.datastructures import CoarsePseudoQuadMesh
from Environment_attempt01 import MeshEnvironment
from compas_viewer import Viewer

# Define the initial and terminal meshes
initial_mesh_vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
initial_mesh_faces = [[0, 1, 2, 3]]
initial_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(initial_mesh_vertices, initial_mesh_faces)

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\RL-attempt-01\trial.json'

# Initialize environment
env = MeshEnvironment(initial_mesh, terminal_mesh_json_path)

# Check the environment
check_env(env)

# Define the RL model
model = DQN('MlpPolicy', env, verbose=1)

#Define callbacks to stop training once a reward threshold is reached
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, eval_freq=1000, verbose=1)

# Train the model
model.learn(total_timesteps=10000, callback=eval_callback)

# Save the model
model.save("dqn_mesh")

# Load the model
model = DQN.load("dqn_mesh")

# Evaluate the trained agent
state, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(state)
    state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

# Visualize the final mesh
view = True
if view:
    viewer = Viewer()
    viewer.scene.add(env.current_mesh)
    viewer.show()
