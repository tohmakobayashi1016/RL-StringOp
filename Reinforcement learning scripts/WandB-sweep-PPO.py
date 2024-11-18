import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from Mesh_Environment_simplified_mesh_mse_A_to_B_final import MeshEnvironment
import sys, os, time, random, torch
import torch.nn.functional as F
import pandas as pd
from compas_quad.datastructures import CoarsePseudoQuadMesh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

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
    

# Step 1: Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'reward',  # Metric to optimize
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # could be higher? for lower bound 1e-2
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512, 1024] # Usually 4 to 4096
        },
        'n_steps': {
            'values': [512, 1024, 2048, 4096, 8192] # AKA Horizon Usually 32 to 5000
        },
        'n_epochs': {
            'values': [3, 5, 10, 20]
        },
        'ent_coef': {
            'values': [0.001, 0.0025, 0.005, 0.0075, 0.01] # Could I try higher than 4 ? carrying over beyond episodes?
        }
    }
}

# Step 2: Define the training function
def train_ppo_with_wandb():
    with wandb.init() as run:
        # Access the sweep parameters
        config = wandb.config

        # Zero state initial mesh
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
        initial_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\at.json'

        # Path to the terminal mesh JSON file
        terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atta.json'

        # Create the environment (replace with your environment)
        env = MeshEnvironment(initial_mesh_json_path, terminal_mesh_json_path, max_steps = 2)

        # Create the model with the specified parameters
        model = PPO(
            'MultiInputPolicy',
            env,
            n_steps=config.n_steps,
            gamma=0.99,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            ent_coef=config.ent_coef,
            learning_rate=config.learning_rate,
            verbose=1
        )

        # Define the number of timesteps for training
        total_timesteps = 10000  # Adjust this based on your needs

        # Train the model with the custom callback for logging
        wandb_callback = WandbMSELoggingCallback()
        model.learn(total_timesteps=total_timesteps*env.max_steps, callback=wandb_callback)

        # Log the final results
        #wandb.log({'reward': model.reward, 'total_timesteps': total_timesteps*env.max_steps})

        # Close the environment
        env.close()

# Step 3: Initialize and run the sweep
if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="Final-A-B-Steps")

    # Start profiling
    start_time = time.time()

    count = 30

    wandb.agent(sweep_id, function=train_ppo_with_wandb, count=count)  # Adjust 'count' as needed

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for {count} counts: {elapsed_time:.2f} seconds")

    # Finish the wandb run
    wandb.finish()

