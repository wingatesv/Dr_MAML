import gym
from gym import spaces
import numpy as np

class MAMLEnv(gym.Env):
    def __init__(self, maml_model):
        super(MAMLEnv, self).__init__()
        self.maml = maml_model
        self.current_epoch = 0

        # Define action space (task_update_num ranging from 1 to 5)
        self.action_space = spaces.Discrete(5)

        # Define observation space (train_loss)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Initial state
        self.state = np.zeros(2)
        
    def reset(self):
        # Reset the MAML model and state for a new episode
        self.current_epoch = 0
        self.state = np.zeros(2)
        print('self.state':, self.state)
        return self.state




# Instantiate environment
# inner_loop_env = InnerLoopEnv(maml_instance)

# # Create PPO agent
# model = PPO("MlpPolicy", inner_loop_env, verbose=1)

# # Instantiate the MAML model and optimizer
# maml_instance = MAML(model_func, n_way, n_support, approx)
# optimizer = torch.optim.Adam(maml_instance.parameters(), lr=0.001)

# # Train PPO agent
# inner_loop_env = InnerLoopEnv(maml_instance)
# ppo_model = PPO("MlpPolicy", inner_loop_env, verbose=1)
# ppo_model.learn(total_timesteps=10000)

# # Train MAML with PPO
# num_epochs = 50  # Define the number of epochs
# for epoch in range(num_epochs):
#     maml_instance.train_loop(epoch, train_loader, optimizer, ppo_model)
#     acc_mean, val_loss = maml_instance.test_loop(test_loader)
#     print(f'Epoch {epoch}: Test Accuracy: {acc_mean:.2f}, Validation Loss: {val_loss:.4f}')
