import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from Mesh_Environment_simplified_hybrid_4 import MeshEnvironment
import sys, os, time, random, torch
import torch.nn.functional as F
import pandas as pd
from compas_quad.datastructures import CoarsePseudoQuadMesh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))


# Import your custom environment and callback here
# Import your custom environment and callback here
class WandbMSELoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbMSELoggingCallback, self).__init__(verbose)
        self.q_network = None
        self.terminal_histogram = None  # To hold the terminal histogram
        self.action_counts = torch.zeros(3, dtype=torch.int32)  # To track actions (0, 1, 2)
        self.losses = []
        self.episode_rewards = []

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
        env = self.training_env.envs[0].unwrapped
        obs = env.get_state()[0]
        info = self.locals.get("infos", [{}])[0]
        terminated = self.locals.get("dones", [False])[0]
        truncated = info.get("TimeLimit.truncated", False)
        current_step = self.training_env.envs[0].unwrapped.current_step
        max_steps = self.training_env.envs[0].unwrapped.max_steps
        loss = self.model.mean_loss

        # Preprare the input as a dictionary of tensors with a batch dimension 
        obs_input = {
            'degree_histogram': torch.tensor([obs['degree_histogram']], dtype=torch.float32),
            'lizard_position': torch.tensor([obs['lizard_position']], dtype=torch.int32)
        }
        
        # Compute the Q-values from the Q-network
        with torch.no_grad():
            q_values = self.q_network(obs_input)

        # Store the loss
        if loss is not None:
            self.losses.append(loss)
            wandb.log({'q_network_loss': loss})

        # Log the Q-values for each action
        wandb.log({
            'Q_value_action_0': q_values[0][0].item(),
            'Q_value_action_1': q_values[0][1].item(),
            'Q_value_action_2': q_values[0][2].item(),
        })


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
            'exploration_rate': self.model.exploration_rate,
        })

        # If episode ends, log the mean reward
        if terminated or current_step == max_steps:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            wandb.log({'mean_reward': mean_reward})
            self.episode_rewards = []  # Reset for next episode

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

# from your_project import  QNetworkLossLoggingCallback

# Step 1: Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'q_network_loss',  # Metric to optimize
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-2, 5e-2, 1e-3, 5e-3, 1e-4] # could be higher? for lower bound 1e-2
        },
        'buffer_size': {
            'values': [1000, 2500, 5000, 7500, 10000]
        },
        'target_update_interval': {
            'min': 500, 
            'max': 10000
        },
        'train_freq': {
            'values': [1, 2, 3, 4, 5, 6, 7, 8] # Could I try higher than 4 ? carrying over beyond episodes?
        }
    }
}

# Step 2: Define the training function
def train_dqn_with_wandb():
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
        initial_mesh = coarse.dense_mesh()
        initial_mesh.collect_strips()
        
        # Path to initial mesh JSON file
        #initial_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\a.json'

        # Path to the terminal mesh JSON file
        terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\meaningful\atpttpta.json'

        # Create the environment (replace with your environment)
        env = MeshEnvironment(initial_mesh, terminal_mesh_json_path, max_steps = 8)

        # Create the model with the specified parameters
        model = DQN(
            'MultiInputPolicy',
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            gamma=0.99,
            target_update_interval=config.target_update_interval,
            train_freq=config.train_freq,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05, 
            verbose=1
        )

        # Define the number of timesteps for training
        total_timesteps = 2000  # Adjust this based on your needs

        # Train the model with the custom callback for logging
        wandb_callback = WandbMSELoggingCallback()
        model.learn(total_timesteps*env.max_steps, callback=wandb_callback)

        # Log the final results
        wandb.log({'final_loss': model.mean_loss, 'total_timesteps': total_timesteps*env.max_steps})

        # Close the environment
        env.close()

# Step 3: Initialize and run the sweep
if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="Final-A-B-Steps")

    # Start profiling
    start_time = time.time()

    # Launch the sweep agentz
    count = 30

    wandb.agent(sweep_id, function=train_dqn_with_wandb, count=count)  # Adjust 'count' as needed

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for {count} counts: {elapsed_time:.2f} seconds")

    # Finish the wandb run
    wandb.finish()

