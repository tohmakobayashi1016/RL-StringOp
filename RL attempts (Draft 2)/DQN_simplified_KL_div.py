from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from compas_quad.datastructures import CoarsePseudoQuadMesh
from Mesh_Environment_simplified_kl_divergence import MeshEnvironment

import wandb, torch
import torch.nn.functional as F

import sys, os, time, random
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Classes.FormatConverter import FormatConverter

# Initialize wandb project
wandb.init(project="DQN-LOSS-FUNCTION", 
           config={
               "policy=type": "MultiInputPolicy",
               "total_timesteps": 100000,
               "env_name": "MeshEnvironment"},
           sync_tensorboard=True,
           save_code=True,
           entity="string-rl")

class WandbLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.q_network = None
        self.terminal_histogram = None  # To hold the terminal histogram
        self.action_counts = torch.zeros(3, dtype=torch.int32)  # To track actions (0, 1, 2)

    def _on_training_start(self) -> None:
        # Access the Q-network for logging Q-values
        self.q_network = self.model.q_net
        
        # Extract terminal_histogram from the environment
        terminal_histogram_dict = self.training_env.envs[0].unwrapped.terminal_histogram
        
        # Convert the terminal histogram dictionary values to a tensor
        self.terminal_histogram = torch.tensor([
            len(terminal_histogram_dict['degree_2_vertices']),
            len(terminal_histogram_dict['degree_3_vertices']),
            len(terminal_histogram_dict['degree_4_vertices']),
            len(terminal_histogram_dict['degree_5_vertices']),
            len(terminal_histogram_dict['degree_6_plus_vertices'])
        ], dtype=torch.float32)

    def _on_step(self) -> bool:
        # Get the current observation from the environment
        obs = self.training_env.envs[0].unwrapped.get_state()[0]

        # Prepare the input as a dictionary of tensors with a batch dimension
        obs_input = {
            'degree_histogram': torch.tensor([obs['degree_histogram']], dtype=torch.float32),
            'lizard_position': torch.tensor([obs['lizard_position']], dtype=torch.int32)
        }

        # Compute the Q-values from the Q-network
        with torch.no_grad():
            q_values = self.q_network(obs_input)

        # Log the observation data (degree histograms, Levenshtein distance, mesh distance)
        wandb.log({
            'degree_2': obs['degree_histogram'][0],
            'degree_3': obs['degree_histogram'][1],
            'degree_4': obs['degree_histogram'][2],
            'degree_5': obs['degree_histogram'][3],
            'degree_6_plus': obs['degree_histogram'][4],
            'lizard_tail': obs['lizard_position'][0],
            'lizard_body': obs['lizard_position'][1],
            'lizard_head': obs['lizard_position'][2],
        })

        # Convert the current degree histogram to tensor and normalize
        current_histogram = torch.tensor(obs['degree_histogram'], dtype=torch.float32)
        epsilon = 1e-10
        current_distribution = (current_histogram + epsilon) / (current_histogram.sum() + epsilon)  # Normalize to get probabilities
        terminal_distribution = (self.terminal_histogram + epsilon) / (self.terminal_histogram.sum() + epsilon)
        # Compute KL-Divergence using torch's F.kl_div function
        kl_divergence = F.kl_div(current_distribution.log(), terminal_distribution, reduction='batchmean')

        # Log the cross-entropy loss along with Q-values, reward, exploration rate, and action distribution
        wandb.log({
            'q_values_mean': q_values.mean().item(),
            'episode_reward': self.locals['rewards'][0] if isinstance(self.locals['rewards'], list) else self.locals['rewards'],
            'exploration_rate': self.model.exploration_rate,
            'q_network_loss': self.locals.get('loss', 0),
            'kl_divergence': kl_divergence.item()  # Log the cross-entropy loss
        })

        # Log action counts (cumulative histogram)
        action = self.locals.get("actions", None)
        if action is not None:
            action = int(action[0])  # Assuming it's a single integer action
            self.action_counts[action] += 1  # Increment action count for the chosen action

            # Log the cumulative count of actions as a histogram
            wandb.log({
                'action_0_count': self.action_counts[0].item(),
                'action_1_count': self.action_counts[1].item(),
                'action_2_count': self.action_counts[2].item(),
            })

        return True
    
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
        lizard_position = obs['lizard_position'].tolist()

        # Append to a list for the current episode
        self.observations.append({
            'degree_histogram': degree_histogram,
            'lizard_position': lizard_position,
            'action': action[0],  # Take the first element if it's a list
            'actions': ''.join(self.actions),
            'reward': reward[0]  # Take the first element if it's a list
        })

        # If the episode is done, log the results
        if done and done[0]:  # 'done' is likely a list, check the first element
            with open(self.log_file, 'a') as f:
                for step_data in self.observations:
                    f.write(f"{self.episode_length},"
                            f"{step_data['degree_histogram'][0]},"
                            f"{step_data['degree_histogram'][1]},"
                            f"{step_data['degree_histogram'][2]},"
                            f"{step_data['degree_histogram'][3]},"
                            f"{step_data['degree_histogram'][4]},"
                            f"{step_data['lizard_position']},"
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
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atpta.json'

# Initialize environment
env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps = 8)
check_env(env)

# Define the RL model
model = DQN('MultiInputPolicy',
            env,
            verbose=1,
            exploration_fraction=0.5,
            exploration_initial_eps=0.9,
            exploration_final_eps=0.1,
            buffer_size=100000)

# Log file path
log_file = 'observation_history_KL_ATPTA_5_2.csv'
episode_log_file = 'episode_rewards_KL_ATPTA_5_2.csv'

# Initialize the log file (with headers)
with open(log_file, 'w') as f:
    f.write('step,degree_2,degree_3,degree_4,degree_5,degree_6_plus,levenshtein_distance,mesh_distance,action,action_letters,reward\n')

with open(episode_log_file, 'w') as f:
    f.write('episode_length,episode_reward,action_string\n')

# Initialize callbacks
num_design_episodes = 1000
observation_callback = ObservationCallback(log_file=log_file)
stop_callback = StopTrainingOnEpisodesCallback(num_episodes=num_design_episodes)
eval_callback = EvalCallback(env, best_model_save_path="./best_agent_performance/", eval_freq=500, deterministic=True, render=False)
episode_logging_callback = EpisodeLoggingCallback(log_file=episode_log_file)
wandb_callback = WandbLoggingCallback()

# Combine callbacks
callback = CallbackList([observation_callback, episode_logging_callback, stop_callback, eval_callback, wandb_callback])

# Start profiling
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
model.save("DQN_simplified_model_KL_ATPTA_5_2")

# Finish the wandb run
wandb.finish()
