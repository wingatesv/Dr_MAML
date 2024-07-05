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
import math

class PPO_MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False,  agent_chkpt_dir = None, test_mode=False):
        super(PPO_MAML, self).__init__(model_func, n_way, n_support, change_way=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task = 4
        self.task_update_num = 5 if test_mode else 3
        self.train_lr = 0.01
        self.approx = approx
        self.test_mode = test_mode
        
        self.action_space = spaces.Discrete(3)
        self.number_of_observations = 2
        # self.number_of_observations = 3
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.number_of_observations,), dtype=np.float32)

        # Setup Agent
        self.agent = Agent(n_actions=self.action_space.n,
                           input_dims=self.observation_space.shape,
                           chkpt_dir = agent_chkpt_dir,
                           alpha=0.01,
                           batch_size= 5, 
                           n_epochs=5, 
                           fc_dims=32 )
        
        self.n_steps = 0
        self.score_history = []
        self.learn_iters = 0
        self.observation = np.full(self.number_of_observations, -1)

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
        loss_all = []
        
        
        self.action, self.prob, self.val = self.agent.choose_action(self.observation)


        self.task_update_num = self.action + 1 # agent.step

      
        print('task_update_num:', self.task_update_num)

        self.n_steps += 1

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

        classifier_weights = list(self.classifier.parameters())[-2]
        weights_mean = torch.mean(classifier_weights.data).detach().cpu().numpy()
        weights_variance = torch.var(classifier_weights.data).detach().cpu().numpy()

        # output_layer_weights.var().cpu().data,
        self.observation = np.array([avg_loss/len(train_loader), epoch])
        # self.observation = np.array([self.task_update_num, avg_loss/len(train_loader)])
        # print('observation: ', self.observation)

    def test_loop(self, test_loader, return_std=False):
        correct = 0
        count = 0
        avg_loss = 0
        done = True
        N = 5
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

        
        if not self.test_mode:
            amp_reward = math.pow(64, (acc_mean / 100) - 1) 
            reward = np.array([acc_mean/100])
            self.agent.remember(self.observation, self.action, self.prob, self.val, reward, done)

            if self.n_steps % N == 0:
                self.agent.learn()
                self.learn_iters +=1
                print('Learn iteration: ', self.learn_iters)
        
        if return_std:
            return acc_mean, acc_std, float(avg_loss / iter_num)
        else:
            return acc_mean, float(avg_loss / iter_num)
