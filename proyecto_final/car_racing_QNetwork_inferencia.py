import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from plot_values import PlotValues

# Q-network and CNN architecture
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(10 * 10 * 32, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 10 * 10 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(),models_path)

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))

# File path
file_path = pathlib.Path(__file__).parent.absolute()

# Initializing the environment
render_mode = "human"
env = gym.make('CarRacing-v2', render_mode=render_mode, continuous=False)
q_network = QNetwork(5)
q_network.load_model('best_model_500.pth')

# Graphic plotter and storing values
plotter = PlotValues()

def run(run_time):
    state = env.reset()[0]
    race_reward = 0

    for t in range(run_time*50):
        if render_mode == "human":
            env.render()
        
        # State preprocess
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Action
        action = torch.argmax(q_network(state_tensor)).numpy()
                
        # Environment action
        next_state, reward, done, truncated, info = env.step(action)
        
        race_reward += reward
        state = next_state
        
        if done or t == run_time*50 - 1:
            print('Total Race Reward:', race_reward)
            break
    
    # Ending the environment
    env.close()

if __name__=="__main__":
    simulation_time = 25 # seconds
    run(simulation_time)