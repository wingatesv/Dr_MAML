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

        # Define observation space (train_loss and val_loss)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        # Initial state
        self.state = np.zeros(2)
        
    def reset(self):
        # Reset the MAML model and state for a new episode
        self.current_epoch = 0
        self.state = np.zeros(2)
        print('self.state':, self.state)
        return self.state

    def step(self, action, train_loss, val_loss):
        # Perform the action by setting the task_update_num in MAML
        self.maml.task_update_num = action + 1

        # Update state with new train and validation losses
        self.state = np.array([train_loss, val_loss])

        # Define the reward as negative validation loss to minimize it
        reward = -val_loss

        # Increment epoch count
        self.current_epoch += 1

        # Check if the episode is done (for simplicity, let's say an episode is 10 epochs)
        done = self.current_epoch >= 10

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Print current state
        print(f'Epoch: {self.current_epoch}, Train Loss: {self.state[0]}, Val Loss: {self.state[1]}')

    def close(self):
        pass



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
