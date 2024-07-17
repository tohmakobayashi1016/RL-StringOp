import numpy as np
import random
from compas_quad.datastructures import CoarsePseudoQuadMesh
from compas.datastructures import Mesh
from compas_quad.grammar.addition2 import lizard_atp
from Environment_attempt01 import MeshEnvironment
import json
import pickle
from compas_viewer import Viewer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}
        self.training_rewards = []
    
    def get_state_action_key(self, state, action):
        return str(state), action
    
    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(self.env.actions)
        else:
            state_key = str(state)
            if state_key not in self.q_table:
                return random.choice(self.env.actions)
            return max(self.env.actions, key=lambda action: self.q_table.get(self.get_state_action_key(state, action), 0))
    
    def update_q_table(self, state, action, reward, next_state):
        state_key = str(state)
        next_state_key = str(next_state)
        state_action_key = self.get_state_action_key(state, action)
        
        if state_action_key not in self.q_table:
            self.q_table[state_action_key] = 0
        
        future_rewards = [self.q_table.get(self.get_state_action_key(next_state, next_action), 0) for next_action in self.env.actions]
        max_future_reward = max(future_rewards)
        
        self.q_table[state_action_key] += self.learning_rate * (reward + self.discount_factor * max_future_reward - self.q_table[state_action_key])
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < self.env.max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.exploration_rate *= self.exploration_decay
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        with open('q_table.pkl','wb') as f:
            pickle.dump(self.q_table, f)
        print("Training complete. Q-table saved to 'q-table.pkl'.")

    def load_q_table(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print("Q-table loaded from", filepath)


# Define the initial and terminal meshes
initial_mesh_vertices = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]
initial_mesh_faces = [[0, 1, 2, 3]]
initial_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(initial_mesh_vertices, initial_mesh_faces)

# The terminal mesh needs to be defined according to your specific goal
# For now, it's a placeholder and should be defined properly
terminal_mesh_json_path = r'C:\Users\footb\Desktop\Thesis\String-RL\Output\RL-attempt-01\trial.json'

env = MeshEnvironment(initial_mesh, terminal_mesh_json_path)

state=env.reset()
print("Initial state:", state)
agent = QLearningAgent(env)

# Train the agent
agent.train(episodes=1000)

# Save training rewards to analyze the training process
with open('training_rewards.pkl', 'wb') as f:
    pickle.dump(agent.training_rewards, f)
print("Training rewards saved to 'training_rewards.pkl'.")

# Evaluate the trained agent
state = env.reset()
done = False

while not done:
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")

# Visualize the final mesh
view = True
if view:
    viewer = Viewer()
    viewer.scene.add(env.current_mesh)
    viewer.show()