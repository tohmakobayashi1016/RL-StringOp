from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList

from compas_quad.datastructures import CoarsePseudoQuadMesh
from Mesh_Environment_simplified import MeshEnvironment

import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Classes.FormatConverter import FormatConverter

class LoggingCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.levenshtein_distances = []
        self.mesh_distances = []
        self.actions = []
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_number = 0

    def _on_step(self) -> bool:
        action = self.locals["actions"]
        reward = self.locals["rewards"]
        done = self.locals["dones"]

        #Accumulate rewards and episode length
        self.episode_reward += float(reward)
        self.episode_length += 1

        #Record action string
        self.format_converter = FormatConverter()
        action_letter = self.training_env.envs[0].format_converter.from_discrete_to_letter([int(action)])
        self.actions.append(action_letter)

        #Collect observation metrics
        obs = self.training_env.envs[0].get_state()[0]
        levenshtein_distance = obs['levenshtein_distance'][0]
        mesh_distance = obs['mesh_distance'][0]

        self.levenshtein_distances.append(levenshtein_distance)
        self.mesh_distances.append(mesh_distance)

        #Log at the end of an episode
        if done:
            self.episode_number += 1
            print(f"Episode {self.episode_number} finished.")
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)

            #Write the log to the file
            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_reward},{self.episode_length},{''.join(self.actions)},{levenshtein_distance},{mesh_distance}\n")
            
            # Reset counters for the next episode
            self.episode_reward = 0
            self.episode_length = 0
            self.actions = []
        
        return True
    
class StopTrainingOnEpisodesCallback(BaseCallback):
    def __init__(self, num_episodes: int, verbose=0):
        super(StopTrainingOnEpisodesCallback, self).__init__(verbose) #Maybe I can also checkout StopTrainingOnRewardCallBack
        self.num_episodes = num_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        #Check if the episode is done
        done = self.locals["dones"][0]
        if done:
            self.episode_count += 1
            print(f"Episode {self.episode_count} finished.")
            if self.episode_count >= self.num_episodes:
                print(f"Stopping training affter {self.episode_count} episodes.")
                return False
        
        return True
    
initial_mesh_vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
initial_mesh_faces = [[0, 1, 2, 3]]
initial_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(initial_mesh_vertices, initial_mesh_faces)

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

# Initialize environment
env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps = 5)
check_env(env)

# Define the RL model
model = DQN('MultiInputPolicy',
            env,
            verbose=1,
            exploration_fraction=0.2,
            exploration_initial_eps=0.9,
            exploration_final_eps=0.1)

# Log file path
log_file = 'training_log_disregard.csv'

# Initialize the log file (with headers)
with open(log_file, 'w') as f:
    f.write('reward,length,actions,levenshtein_distance,mesh_distance\n')

# Initialize callbacks
num_design_episodes = 1000
logging_callback = LoggingCallback(log_file=log_file)
stop_callback = StopTrainingOnEpisodesCallback(num_episodes=num_design_episodes)
reward_callback = StopTrainingOnRewardThreshold(reward_threshold=0.9, verbose=1) #check implementation
eval_callback = EvalCallback(env, callback_on_new_best=reward_callback, eval_freq=1000, verbose=1) #check implementation

# Combine callbacks
callback = CallbackList([logging_callback, stop_callback, reward_callback, eval_callback])

#Start profiling
start_time = time.time()

# Start training
model.learn(total_timesteps=num_design_episodes * env.max_steps,
            log_interval=4,
            reset_num_timesteps=False,
            callback=callback)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time for {num_design_episodes} episodes: {elapsed_time:.2f} seconds")

# Save the model
model.save("dqn_mesh_graph_disregard")


