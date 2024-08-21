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

        self.train_lr = 0.01
        self.approx = approx
        self.test_mode = test_mode
        
        self.action_space = spaces.Discrete(2)
        self.number_of_observations = 3

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.number_of_observations,), dtype=np.float32)

        # Setup Agent
        self.agent = Agent(n_actions=self.action_space.n,
                           input_dims=self.observation_space.shape,
                           chkpt_dir = agent_chkpt_dir,
                           alpha=0.01,
                           batch_size= 5, 
                           fc_dims=32 )
      

        self.n_steps = 0
        self.task_update_num = 0
        self.done = False

        self.learn_iters = 0
        self.reward_history = []
        self.observation = np.full(self.number_of_observations, -1)


        # Store metrics for plotting
    #     self.metrics = {
  
    #         'average_support_loss': [],
    #         'query_loss': [],
    #         'task_update_num': [],
    #         'reward': [],
    #     }
        
   
    # def collect_metrics(self, reward, acc_mean):
    #     self.metrics['average_support_loss'].append(self.average_support_loss)
    #     self.metrics['query_loss'].append(self.query_loss)
    #     self.metrics['task_update_num'].append(self.task_update_num)
    #     self.metrics['reward'].append(reward)
    #     print(f"Metrics collected for epoch {self.current_epoch}.")
        
    def reset_environment(self):
        self.n_steps = 0 
        self.task_update_num = 0
        self.done = False

        
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


        
        while self.task_update_num < self.max_task_update_num:
            # Forward pass on support set
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i)
            self.support_loss = set_loss.item()
            
            # Compute gradients with respect to fast parameters
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
            if self.approx:
                grad = [g.detach() for g in grad]
            
            # Compute gradient norm
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grad]))
            self.grad_norm = grad_norm.item()
            
            # Create observation vector for the PPO agent
            self.observation = np.array([self.support_loss, self.grad_norm, self.task_update_num])
            print('Observation: ', self.observation)
    
            # PPO agent decides whether to continue adaptation or stop
            self.action, self.prob, self.val = self.agent.choose_action(self.observation)
            print('PPO Action:', self.action)
            self.n_steps += 1
            print('Number of Iteration:', self.n_steps)

            if not self.test_mode:
                # Calculate reward (example: negative support loss to maximize improvement)
                reward = -self.support_loss  # Reward could also include other components
                self.reward_history.append(reward)
            
                # Determine if the episode is done
                self.done = (self.action == 0) or (self.n_steps + 1 >= max_steps)
                
                # Store the experience in the PPO agent's memory
                self.agent.remember(self.observation, self.action, self.prob, self.val, reward, self.done)
            
            # If PPO agent decides to stop, break the loop
            if self.action == 0:  # Action '0' means stop adaptation
                break
            
            # Perform the adaptation step
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                fast_parameters.append(weight.fast)
            
           
            self.task_update_num += 1
            
      
        print('Total adaptation step:',self.task_update_num)
        scores = self.forward(x_b_i)
        # Learn from the episode if it has ended
        if self.done and not self.test_mode:
            self.agent.learn(n_epochs=5)
            self.learn_iters +=1
            print('Learn iteration: ', self.learn_iters)
            # Collect metrics at the end of each epoch
            # self.collect_metrics(reward)
            #reset environment
            self.reset_environment()
            
        return scores

    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature=False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        self.query_loss = loss.item()
        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []

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
                optimizer.step()

                task_count = 0
                loss_all = []
            optimizer.zero_grad()
        
            
            if i % print_freq == 0:
                print(f'Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss {avg_loss / float(i + 1):.6f}')

        print(f'Epoch {epoch} | Batch {len(train_loader)}/{len(train_loader)} | Avg Loss {(avg_loss / len(train_loader)):.6f}')



    def calculate_reward(self):
        reward = 0
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
