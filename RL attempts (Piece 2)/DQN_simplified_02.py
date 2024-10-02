from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CallbackList

from compas_quad.datastructures import CoarsePseudoQuadMesh
from Mesh_Environment_simplified import MeshEnvironment

import sys, os, time
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Classes.FormatConverter import FormatConverter

class ObservationCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super(ObservationCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.observations = []
        self.actions = []
        self.episode_reward = 0
        self.episode_length = 0
        self.format_converter = FormatConverter()

    def _on_step(self) -> bool:
        # Try to access 'actions' and 'rewards' safely from self.locals
        action = self.locals.get("actions", None)  # Note the plural 'actions'
        reward = self.locals.get("rewards", None)  # Note the plural 'rewards'
        done = self.locals.get("dones", None)  # Note the plural 'dones'

        # Ensure 'actions' and 'rewards' are available before proceeding
        if action is None or reward is None:
            print("Warning: 'actions' or 'rewards' not available in self.locals")
            return True  # Return True to avoid stopping training

        # Convert action to action letter using FormatConverter
        action_letter = self.training_env.envs[0].format_converter.from_discrete_to_letter([int(action)])
        self.actions.append(action_letter)

        # Accumulate rewards and episode length
        self.episode_reward += reward[0]  # It's likely a list, take the first element
        self.episode_length += 1

        # Log observation data
        obs = self.training_env.envs[0].get_state()[0]
        degree_histogram = obs['degree_histogram'].tolist()
        levenshtein_distance = obs['levenshtein_distance'][0]
        mesh_distance = obs['mesh_distance'][0]

        # Extract degree histogram values without brackets
        degree_2 = degree_histogram[0]
        degree_3 = degree_histogram[1]
        degree_4 = degree_histogram[2]
        degree_5 = degree_histogram[3]
        degree_6_plus = degree_histogram[4]

        # Append to a list for the current episode
        self.observations.append({
            'degree_2': degree_2,
            'degree_3': degree_3,
            'degree_4': degree_4,
            'degree_5': degree_5,
            'degree_6_plus': degree_6_plus,
            'levenshtein_distance': levenshtein_distance,
            'mesh_distance': mesh_distance,
            'action': action[0],  # Take the first element if it's a list
            'actions': ''.join(self.actions),
            'reward': reward[0]  # Take the first element if it's a list
        })

        # If the episode is done, log the results
        if done and done[0]:  # 'done' is likely a list, check the first element
            with open(self.log_file, 'a') as f:
                for step_data in self.observations:
                    f.write(f"{self.episode_length},"
                            f"{step_data['degree_2']},"
                            f"{step_data['degree_3']},"
                            f"{step_data['degree_4']},"
                            f"{step_data['degree_5']},"
                            f"{step_data['degree_6_plus']},"
                            f"{step_data['levenshtein_distance']},"
                            f"{step_data['mesh_distance']}," 
                            f"{step_data['action']}," 
                            f"{step_data['actions']},"
                            f"{step_data['reward']}\n")

            # Reset for the next episode
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0
            self.actions = []
            self.observations = []  # Clear observations for the next episode

        return True

class EpisodeLoggingCallback(BaseCallback):
    def __init__(self, log_file, verbose=0):
        super(EpisodeLoggingCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_strings = []  # To store the full action string at the end of each episode
        self.actions = []  # List to store actions taken in the current episode
        self.episode_reward = 0
        self.episode_length = 0
        self.format_converter = FormatConverter()

    def _on_step(self) -> bool:
        # Access actions, rewards, and done flags from self.locals
        action = self.locals.get("actions", None)
        reward = self.locals.get("rewards", None)
        done = self.locals.get("dones", None)

        # Ensure actions and rewards are available
        if action is None or reward is None:
            print("Warning: 'actions' or 'rewards' not available in self.locals")
            return True  # Continue training

        # Convert action to action letter and append to the actions list
        action_letter = self.training_env.envs[0].format_converter.from_discrete_to_letter([int(action[0])])
        self.actions.append(action_letter)  # Append to the current action list

        # Accumulate reward and episode length
        self.episode_reward += reward[0]
        self.episode_length += 1

        # If the episode is done, log the results
        if done[0]:
            # Join the list of actions to create the full action string
            full_action_string = ''.join(self.actions)
            with open(self.log_file, 'a') as f:
                # Log episode length, reward, full action string, and the length of the action string
                f.write(f"{self.episode_length},{self.episode_reward},{full_action_string}\n")

            # Reset for the next episode
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self.episode_strings.append(full_action_string)  # Store the final action string
            self.episode_reward = 0
            self.episode_length = 0
            self.actions = []  # Reset the actions list for the next episode

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
    
input_mesh_refinement = 2  # densify the input 1-face quad mesh

# dummy mesh with a single quad face
vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
faces = [[0, 1, 2, 3]]
coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

# denser mesh
coarse.collect_strips()
coarse.strips_density(input_mesh_refinement)
coarse.densification()
initial_mesh = coarse.dense_mesh()
initial_mesh.collect_strips()

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
log_file = 'observation_history.csv'
episode_log_file = 'episode_rewards_log.csv'

# Initialize the log file (with headers)
with open(log_file, 'w') as f:
    f.write('step,degree_2,degree_3,degree_4,degree_5,degree_6_plus,levenshtein_distance,mesh_distance,action,action_letters,reward\n')

with open(episode_log_file, 'w') as f:
    f.write('episode_length,episode_reward,action_string\n')

# Initialize callbacks
num_design_episodes = 10000
observation_callback = ObservationCallback(log_file=log_file)
stop_callback = StopTrainingOnEpisodesCallback(num_episodes=num_design_episodes)
reward_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1) #check implementation
eval_callback = EvalCallback(env, callback_on_new_best=reward_callback, eval_freq=1000, verbose=1) #check implementation
episode_logging_callback = EpisodeLoggingCallback(log_file=episode_log_file)

# Combine callbacks
callback = CallbackList([observation_callback, episode_logging_callback, stop_callback, reward_callback, eval_callback])

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


