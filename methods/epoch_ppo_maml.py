# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
from methods.ppo_torch import Agent
from gym import spaces
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt

import math

class PPO_MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False,  agent_chkpt_dir = None, test_mode=False):
        super(PPO_MAML, self).__init__(model_func, n_way, n_support, change_way=False)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task = 4
        self.max_task_update_num = 5
        self.task_update_num = 5 if test_mode else 3
        self.train_lr = 0.01
        self.approx = approx
        self.test_mode = test_mode
        
        self.action_space = spaces.Discrete(3)
        self.number_of_observations = 6

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.number_of_observations,), dtype=np.float32)

        # Setup Agent
        self.agent = Agent(n_actions=self.action_space.n,
                           input_dims=self.observation_space.shape,
                           chkpt_dir = agent_chkpt_dir,
                           alpha=0.01,
                           batch_size= 5, 
                           fc_dims=64 )

        self.n_steps = 0
        self.done = False
        self.num_cycle = 20
        self.learn_iters = 0
        self.reward_history = []
        self.observation = np.full(self.number_of_observations, -1)

        self.previous_train_loss = 0
        self.previous_val_loss = 0

        # Store metrics for plotting
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'grad_norm': [],
            'task_update_num': [],
            'reward': [],
            'acc_mean': []
        }
        
   
    def collect_metrics(self, reward, acc_mean):
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['train_loss'].append(self.train_loss)
        self.metrics['val_loss'].append(self.val_loss)
        self.metrics['grad_norm'].append(self.grad_norm)
        self.metrics['task_update_num'].append(self.task_update_num)
        self.metrics['reward'].append(reward)
        self.metrics['acc_mean'].append(acc_mean/100)
        print(f"Metrics collected for epoch {self.current_epoch}.")
        
    def reset_environment(self):
        self.n_steps = 0 
        self.done = False
        self.previous_train_loss = self.train_loss
        self.previous_val_loss = self.val_loss
        self.reward_history = []
        self.task_update_num = 3
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, is_feature=False):
        assert not is_feature, 'MAML does not support fixed feature'
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support, *x.size()[2:]) 
        x_b_i = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query, *x.size()[2:]) 
        y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()
        
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
            if self.approx:
                grad = [g.detach() for g in grad]
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                fast_parameters.append(weight.fast)

        scores = self.forward(x_b_i)
        return scores

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        task_count = 0
        self.grad_norm = 0.0
        loss_all = []
        

        self.set_epoch(epoch)
        
        self.action, self.prob, self.val = self.agent.choose_action(self.observation)

        # Apply action to task_update_num
        if self.action == 0:
            self.task_update_num = max(1, self.task_update_num - 1)
        elif self.action == 1:
            self.task_update_num = min(5, self.task_update_num + 1)
            
        print('task_update_num:', self.task_update_num)
        self.n_steps += 1
        print('Number of Iteration:',  self.n_steps)

        optimizer.zero_grad()
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"


            loss = self.set_forward_loss(x)
            avg_loss += loss.item()
            loss_all.append(loss)

            task_count += 1
            if task_count == self.n_task:
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()

                # Calculate gradient norm
                for param in self.parameters():
                    if param.grad is not None:
                        self.grad_norm += param.grad.norm().item() ** 2
                self.grad_norm = self.grad_norm ** 0.5  # Take the square root to get the norm
    
                
                optimizer.step()

                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print(f'Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss {avg_loss / float(i + 1):.6f}')

        print(f'Epoch {epoch} | Batch {len(train_loader)}/{len(train_loader)} | Avg Loss {(avg_loss / len(train_loader)):.6f}')

        self.train_loss = avg_loss/len(train_loader)

    def calculate_energy_based_reward(self, previous_train_loss, train_loss, previous_val_loss, val_loss, gd_steps, max_gd_steps):
        total_epochs = 200
        # Initialize histories if not already present
        if not hasattr(self, 'train_loss_history'):
            self.train_loss_history = []
        if not hasattr(self, 'val_loss_history'):
            self.val_loss_history = []
            
        # Track efficiency history for smoothing (assuming these are lists initialized elsewhere)
        if not hasattr(self, 'train_efficiency_history'):
            self.train_efficiency_history = []
        if not hasattr(self, 'val_efficiency_history'):
            self.val_efficiency_history = []
            
        # Append current losses to history
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        
        # Calculate the relative efficiency as the percentage change in loss per GD step
        train_efficiency = ((previous_train_loss - train_loss) / previous_train_loss) / gd_steps if gd_steps > 0 and previous_train_loss > 0 else 0
        val_efficiency = ((previous_val_loss - val_loss) / previous_val_loss) / gd_steps if gd_steps > 0 and previous_val_loss > 0 else 0
        
        # Apply a diminishing returns adjustment to GD steps
        adjusted_gd_steps = gd_steps ** 0.5  # Square root to reduce the impact of higher steps
        
        # Calculate adjusted efficiencies with diminishing returns factored in
        train_efficiency_adjusted = (previous_train_loss - train_loss) / adjusted_gd_steps if adjusted_gd_steps > 0 else 0
        val_efficiency_adjusted = (previous_val_loss - val_loss) / adjusted_gd_steps if adjusted_gd_steps > 0 else 0
        
        # Store efficiencies for smoothing
        self.train_efficiency_history.append(train_efficiency_adjusted)
        self.val_efficiency_history.append(val_efficiency_adjusted)
        
        # Smooth the efficiencies over the last few epochs (e.g., last 3 epochs)
        smoothed_train_efficiency = np.mean(self.train_efficiency_history[-3:])
        smoothed_val_efficiency = np.mean(self.val_efficiency_history[-3:])
        
        # Dynamically weight the importance of training vs. validation efficiency
        weight = self.current_epoch / total_epochs  # Early epochs favor training, later favor validation
        efficiency = (1 - weight) * smoothed_train_efficiency + weight * smoothed_val_efficiency
        print('Smoothed Efficiency: ', round(efficiency, 4))
      
        # Loss convergence and trend detection using linear regression over last few epochs
        trend_window = 5  # Number of epochs to consider for trend detection
        if len(self.train_loss_history) >= trend_window:
            train_slope, _, _, _, _ = linregress(range(trend_window), self.train_loss_history[-trend_window:])
            val_slope, _, _, _, _ = linregress(range(trend_window), self.val_loss_history[-trend_window:])
            
            if train_slope < 0 and val_slope < 0:
                convergence_bonus = 0.2  # Both losses trending downwards
                print('Convergence trend detected, adding bonus.')
            elif train_slope < 0 and val_slope >= 0:
                convergence_bonus = -0.2  # Training loss decreasing but validation loss not, potential overfitting
                print('Divergence trend detected, applying penalty.')
            else:
                convergence_bonus = 0  # No significant trend detected
                print('No significant trend detected.')
        else:
            convergence_bonus = 0  # Not enough data to detect a trend
            print('Not enough epochs to detect trend, no convergence bonus applied.')
        
        # Adaptive penalty based on performance
        if efficiency > 0.01 and self.grad_norm > 10:  # Good efficiency and high gradient norm suggest not converging yet
            penalty = (gd_steps / max_gd_steps) ** 2 * 0.5  # Reduced penalty
        else:
            penalty = (gd_steps / max_gd_steps) ** 2  # Full penalty if efficiency is low
        print('Penalty: ', round(penalty, 4))
        
        # Exploration bonus, more significant at the beginning of training
        exploration_bonus = 0.2 * (1 - self.current_epoch / total_epochs) if gd_steps >= 4 else 0
        print('Exploration Bonus: ', round(exploration_bonus, 4))
        
        # Final energy-based reward
        reward = (1.5 * efficiency * (1 - penalty)) + convergence_bonus + exploration_bonus
        print('Reward: ', round(reward, 4))
        
        return reward

    
    def test_loop(self, test_loader, return_std=False):
        correct = 0
        count = 0
        avg_loss = 0

        
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"
            correct_this, count_this, loss = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
            avg_loss += loss.item()

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(f'{iter_num} Test Acc = {acc_mean:.2f}% ± {1.96 * acc_std / np.sqrt(iter_num):.2f}%, Test Loss = {avg_loss / iter_num:.4f}')

        self.val_loss = avg_loss/len(test_loader)

        
        
        if not self.test_mode:
    
            self.observation = np.array([self.current_epoch, self.train_loss, self.val_loss, self.grad_norm, self.task_update_num, (acc_mean/100)])
            print('Observation: ', self.observation)

            # Calculate the energy-based reward
            reward = self.calculate_energy_based_reward(
                self.previous_train_loss, self.train_loss,
                self.previous_val_loss, self.val_loss,
                self.task_update_num, self.max_task_update_num
            )
            
            #update previous losses
            self.previous_train_loss = self.train_loss
            self.previous_val_loss = self.val_loss
            self.reward_history.append(reward)

            # Check for reward worsening or plateau (adaptive episode ending)
            if len(self.reward_history) >= int(self.num_cycle/2):
                if reward < self.reward_history[-int(self.num_cycle/2)]:  # Reward worsens
                    self.done = True
                elif len(self.reward_history) >= self.num_cycle and np.std(self.reward_history[-self.num_cycle:]) < 1e-4:  # Improvement plateaus
                    self.done = True
            
            self.agent.remember(self.observation, self.action, self.prob, self.val, reward, self.done)     
         
            if self.done or (self.n_steps+1) % self.num_cycle  == 0:
                self.agent.learn(n_epochs = 5)
                self.learn_iters +=1
                print('Learn iteration: ', self.learn_iters)

                if self.done:
                    print('Episode ended early due to worsening reward or plateau.')
                else:
                     print(f'Episode ended at {self.n_steps+1} cycle')

                #reset environment
                self.reset_environment()
                    
            # Collect metrics at the end of each epoch
            self.collect_metrics(reward, acc_mean)
        
        if return_std:
            return acc_mean, acc_std, float(avg_loss / iter_num)
        else:
            return acc_mean, float(avg_loss / iter_num)

    # New method to output metrics and plot graphs
    def output_metrics(self, file_path=None, save_plots=True, plot_dir='/content/'):
        # Convert the metrics dictionary to a pandas DataFrame
        metrics_df = pd.DataFrame(self.metrics)
        
        if file_path:
            # Save the DataFrame as a CSV file
            metrics_df.to_csv(file_path, index=False)
            print(f"Metrics saved to {file_path}")
            
        # Calculate cumulative reward
        metrics_df['cumulative_reward'] = metrics_df['reward'].cumsum()

        # General plot styling settings
        plt.rcParams.update({
            'font.size': 14,  # Regular font size for ticks
            'axes.labelsize': 16,  # Larger font size for axis labels
            'axes.labelweight': 'bold',  # Bold axis labels
            'lines.linewidth': 2,  # Thicker lines for better visibility
            'legend.fontsize': 14,  # Slightly larger legend text
            'legend.frameon': False,  # No box around legend
            'axes.spines.top': False,  # Remove top spine for a cleaner look
            'axes.spines.right': False,  # Remove right spine for a cleaner look
        })

        # Color scheme for lines
        color_scheme = {
            'train_loss': '#1f77b4',  # Light blue
            'val_loss': '#ff7f0e',    # Light orange
            'acc_mean': '#2ca02c',    # Light green
            'grad_norm': '#9467bd',   # Light purple
            'task_update_num': '#17becf',  # Light cyan
            'reward': '#e377c2',      # Light pink
            'cumulative_reward': '#7f7f7f', # Light gray
        }

        # First plot: Train Loss, Validation Loss, and Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epochs'], metrics_df['train_loss'], label='Train Loss', color=color_scheme['train_loss'])
        plt.plot(metrics_df['epochs'], metrics_df['val_loss'], label='Validation Loss', color=color_scheme['val_loss'])
        plt.plot(metrics_df['epochs'], metrics_df['acc_mean'], label='Accuracy', color=color_scheme['acc_mean'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend(loc='best')
        if save_plots:
            plt.savefig(f'{plot_dir}train_val_acc_plot.png', bbox_inches='tight')
        plt.show()

        # Second plot: Gradient Norm
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epochs'], metrics_df['grad_norm'], label='Gradient Norm', color=color_scheme['grad_norm'])
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend(loc='best')
        if save_plots:
            plt.savefig(f'{plot_dir}grad_norm_plot.png', bbox_inches='tight')
        plt.show()
        
        # Third plot: Task Update Number, Reward, and Cumulative Reward
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epochs'], metrics_df['task_update_num'], label='Task Update Num', color=color_scheme['task_update_num'])
        plt.plot(metrics_df['epochs'], metrics_df['reward'], label='Reward', color=color_scheme['reward'])
        plt.plot(metrics_df['epochs'], metrics_df['cumulative_reward'], label='Cumulative Reward', color=color_scheme['cumulative_reward'])
        plt.xlabel('Epoch')
        plt.ylabel('Task Update Num / Reward')
        plt.legend(loc='best')
        if save_plots:
            plt.savefig(f'{plot_dir}task_update_reward_plot.png', bbox_inches='tight')
        plt.show()

        # Return the DataFrame for further use if needed
        return metrics_df

