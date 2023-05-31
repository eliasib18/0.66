import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib
from plot_values import PlotValues

file_path = pathlib.Path(__file__).parent.absolute()

# File path
file_path = pathlib.Path(__file__).parent.absolute()

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
        # TODO: Carga los pesos de tu red neuronal
        self.load_state_dict(torch.load(models_path))

# Initializing the environment
render_mode = "rgb_array"
    # "human"
    # "rgb_array"
env = gym.make('CarRacing-v2', render_mode=render_mode)

# Set seed for replicable results
np.random.seed(0)
torch.manual_seed(0)

# Number of possible actions (Steer, Accelerate, Brake)
action_size = 3

# Graphic plotter and storing values
plotter = PlotValues()

# Hyperparameters
total_epochs = 100
max_timesteps = 1000
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.0001

# Q-network creation
q_network = QNetwork(action_size)
q_network.save_model('pesos_iniciales.pth')
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Training
for epoch in range(total_epochs):
    state = env.reset()[0] #[0] 96x96x3
    epoch_reward = 0
    
    for t in range(max_timesteps):
        env.render()
        
        # State pre
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # epsilon-greedy exploration
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = q_network(state_tensor).detach().numpy()[0]
        
        # TEnvironment action
        next_state, reward, done, truncated, info = env.step(action)
        
        # Next state preprocess
        next_state_tensor = torch.FloatTensor(next_state).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Bellman equation for Q-values
        q_value = q_network(state_tensor)[0]
        next_q_value = q_network(next_state_tensor).max().detach()
        target_q_value = reward + gamma * next_q_value
        # print('Q Value:', q_value, 'Target Q Value:', target_q_value)
        
        # Optimization
        loss = nn.MSELoss()(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_reward += reward
        state = next_state
        
        if done or t == max_timesteps - 1:
            print('Epoch:', epoch, 'Reward:', epoch_reward, 'Epsilon:', epsilon)
            break
    
    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Plotting Reward and Epsilon values
    plotter.on_epoch_end(total_epochs, epoch_reward)
# Closing Environment
env.close()

# Image save
# plotter.on_train_end("reward_&_epsilon")

# Optimized moodel
q_network.save_model("best_model.pth")