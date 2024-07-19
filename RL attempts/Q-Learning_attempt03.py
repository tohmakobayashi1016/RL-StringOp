from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from compas_quad.datastructures import CoarsePseudoQuadMesh
from Environment_attempt04 import MeshEnvironment
from compas_viewer import Viewer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Classes')))
from FormatConverter import FormatConverter

class LoggingCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.actions = []
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        action = self.locals["actions"]
        reward = self.locals["rewards"]
        done = self.locals["dones"]

        self.episode_reward += float(reward)
        self.episode_length += 1

        self.format_converter = FormatConverter()
        action_letter = self.training_env.envs[0].format_converter.from_discrete_to_letter([int(action)])
        self.actions.append(action_letter)
        
        print(f"Action taken: {action_letter}")

        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_reward},{self.episode_length},{''.join(self.actions)}\n")
            self.episode_reward = 0
            self.episode_length = 0
            self.actions = []
        return True

# Define the initial and terminal meshes
initial_mesh_vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
initial_mesh_faces = [[0, 1, 2, 3]]
initial_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(initial_mesh_vertices, initial_mesh_faces)

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\RL-attempt-01\trial.json'

# Initialize environment
env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps = 1000)

# Check the environment
check_env(env)

# Define the RL model
model = DQN('MlpPolicy', env, verbose=1)

#Define the log file path
log_file = 'training_log.csv'

#Initialize the log file
with open(log_file, 'w') as f:
    f.write('reward, length,actions\n')

#Initialize the callback

Logging_callback = LoggingCallback(log_file)


#Define callbacks to stop training once a reward threshold is reached
#callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1)
#eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, eval_freq=1000, verbose=1)

# Train the model
model.learn(total_timesteps=10000, callback=Logging_callback)

# Save the model
model.save("dqn_mesh")

# Load the model
model = DQN.load("dqn_mesh")

# Evaluate the trained agent
state, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}")

# Visualize the final mesh
view = False
if view:
    viewer = Viewer()
    viewer.scene.add(env.current_mesh)
    viewer.show()
