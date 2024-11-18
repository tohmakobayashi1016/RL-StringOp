from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from compas_quad.datastructures import CoarsePseudoQuadMesh
from Mesh_Environment_simplified_mesh_mse_A_to_B_final import MeshEnvironment

import wandb, torch
import torch.nn.functional as F

import sys, os, time, random
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Classes.FormatConverter import FormatConverter

# Initialize wandb project
wandb.init(project="Final-A-B-Steps", 
           config={
               "policy=type": "MultiInputPolicy",
               "total_timesteps": 100000,
               "env_name": "MeshEnvironment"},
           sync_tensorboard=True,
           save_code=True,
           entity="string-rl")

class WandbMSELoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbMSELoggingCallback, self).__init__(verbose)
        self.policy_network = None
        self.terminal_histogram = None  # To hold the terminal histogram
        self.action_counts = torch.zeros(3, dtype=torch.int32)  # To track actions (0, 1, 2)
        self.losses = []
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self) -> None:        
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
        env = self.training_env.envs[0].unwrapped
        obs = env.get_state()[0]
        info = self.locals.get("infos", [{}])[0]
        terminated = self.locals.get("dones", [False])[0]
        current_step = self.training_env.envs[0].unwrapped.current_step
        max_steps = self.training_env.envs[0].unwrapped.max_steps

        if self.locals.get('infos', None) is not None:
            entropy_loss = self.model.logger.name_to_value['train/entropy_loss']
            policy_loss = self.model.logger.name_to_value['train/policy_loss']
            value_loss = self.model.logger.name_to_value['train/value_loss']

            if isinstance(self.model, PPO):
                kl_divergence = self.model.logger.name_to_value.get('train/approx_kl',0)
                wandb.log({'KL Divergence': kl_divergence})
            
            wandb.log({
                'Policy Loss': policy_loss,
                'Value Loss': value_loss,
                'Entropy': entropy_loss,
                'Steps': self.num_timesteps
            })
        #policy_loss = self.model.policy_loss
        #value_loss = self.model.value_loss
        
        #if policy_loss is not None:
            #self.losses.append(policy_loss)
            #wandb.log({'policy_loss': policy_loss})
        
        #if value_loss is not None:
            #wandb.log({'value_loss': value_loss})

        # Log the observation data (degree histograms, lizard_position)
        if current_step < max_steps and not terminated:
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
        if (current_step == max_steps or terminated) and 'degree_histogram' in info and 'lizard_position' in info:
            wandb.log({
                'degree_2': info['degree_histogram'][0],
                'degree_3': info['degree_histogram'][1],
                'degree_4': info['degree_histogram'][2],
                'degree_5': info['degree_histogram'][3],
                'degree_6_plus': info['degree_histogram'][4],
                'lizard_tail': info['lizard_position'][0],
                'lizard_body': info['lizard_position'][1],
                'lizard_head': info['lizard_position'][2],
                })

        # Convert the current degree histogram to tensor and normalize
        if current_step < max_steps and not terminated:
            current_histogram = torch.tensor(obs['degree_histogram'], dtype=torch.float32)
            mse = F.mse_loss(current_histogram, self.terminal_histogram)
            wandb.log({'mse_loss': mse.item()})
        
        if (current_step == max_steps or terminated) and 'degree_histogram' in info and 'lizard_position' in info:
            current_histogram = torch.tensor(info['degree_histogram'], dtype=torch.float32)
            mse = F.mse_loss(current_histogram, self.terminal_histogram)
            wandb.log({'mse_loss': mse.item()})

        # Collect reward observations
        reward = self.locals['rewards'][0] if isinstance(self.locals['rewards'], list) else self.locals['rewards']
        self.episode_rewards.append(reward)

        # Log reward and exploration rate
        wandb.log({
            'reward': reward,
        })

        # If episode ends, log the mean reward
        if terminated or current_step == max_steps:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            wandb.log({'mean_reward': mean_reward})
            self.episode_rewards = []  # Reset for next episode
            self.episode_lengths.append(current_step)

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
        self.episode_action = []
        self.episode_reward = 0
        self.episode_length = 0
        self.format_converter = FormatConverter()

    def _on_step(self) -> bool:
        # Try to access 'actions' and 'rewards' safely from self.locals
        action = self.locals.get("actions", None)  # Note the plural 'actions'
        reward = self.locals.get("rewards", None)  # Note the plural 'rewards'
        done = self.locals.get("dones", None)  # Note the plural 'dones'
        info = self.locals.get("infos", [{}])[0]
        terminated = self.locals.get("dones", [False])[0]
        
        current_step = self.training_env.envs[0].unwrapped.current_step
        max_steps = self.training_env.envs[0].unwrapped.max_steps

        # Ensure 'actions' and 'rewards' are available before proceeding
        if action is None or reward is None:
            print("Warning: 'actions' or 'rewards' not available in self.locals")
            return True  # Return True to avoid stopping training

        # Convert action to action letter using FormatConverter
        action_letter = self.training_env.envs[0].format_converter.from_discrete_to_letter([int(action)])
        self.actions.append(action_letter)

        # Accumulate rewards and episode length
        self.episode_reward = reward[0]
        self.episode_length += 1

        # Log observation data
    
        if current_step < max_steps and not terminated:
            obs = self.training_env.envs[0].get_state()[0]
            degree_histogram = obs.get('degree_histogram',[])
            lizard_position = obs.get('lizard_position',[])
            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_length},"
                        f"{degree_histogram[0]},"
                        f"{degree_histogram[1]},"
                        f"{degree_histogram[2]},"
                        f"{degree_histogram[3]},"
                        f"{degree_histogram[4]},"
                        f"{lizard_position[0]},"
                        f"{lizard_position[1]},"
                        f"{lizard_position[2]},"
                        f"{action[0]},"  
                        f"{''.join(self.actions)},"
                        f"{reward[0]}\n"  
                )
        if (current_step == max_steps or terminated) and 'degree_histogram' in info and 'lizard_position' in info:
            with open(self.log_file, 'a') as f:
                f.write(f"{self.episode_length},"
                f"{info['degree_histogram'][0]},"
                f"{info['degree_histogram'][1]},"
                f"{info['degree_histogram'][2]},"
                f"{info['degree_histogram'][3]},"
                f"{info['degree_histogram'][4]},"
                f"{info['lizard_position'][0]},"
                f"{info['lizard_position'][1]},"
                f"{info['lizard_position'][2]},"
                f"{action[0]},"
                f"{''.join(self.actions)},"
                f"{reward[0]},"
                f"{''.join(self.actions)}\n"
                )

        if done and done[0]:
            # Reset for the next episode
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0
            self.actions = []
            self.observations = []  # Clear observations for the next episode

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
#initial_mesh = coarse.dense_mesh()
#initial_mesh.collect_strips()

# Path to initial mesh JSON file
initial_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\a.json'

# Path to the terminal mesh JSON file
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

# Initialize environment
env = MeshEnvironment(initial_mesh_json_path, terminal_mesh_json_path, max_steps = 3)
check_env(env)

# Define the RL model
model = PPO('MultiInputPolicy',
            env,
            verbose=1,
            gamma=0.99,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            ent_coef=0.01,
            learning_rate=1e-3)

# Log file path
log_file = 'observation_PPO_mesh_three_step_mse.csv'

# Initialize the log file (with headers)
with open(log_file, 'w') as f:
    f.write('step,degree_2,degree_3,degree_4,degree_5,degree_6_plus,lizard_position,action,action_letters,reward\n')

# Initialize callbacks
num_design_episodes = 1000
observation_callback = ObservationCallback(log_file=log_file)
eval_callback = EvalCallback(env, best_model_save_path="./best_agent_performance/", eval_freq=500, deterministic=True, render=False)
wandb_callback = WandbMSELoggingCallback()

# Combine callbacks
callback = CallbackList([observation_callback, eval_callback, wandb_callback])

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
model.save("PPO_simplified_mesh_three_step_mse")

# Finish the wandb run
wandb.finish()
