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

class PPO_MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False, test_mode=False):
        super(PPO_MAML, self).__init__(model_func, n_way, n_support, change_way=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task = 4
        self.task_update_num = 5 if test_mode else 3
        self.train_lr = 0.01
        self.approx = approx
        self.test_mode = test_mode
        self.inner_loop_steps_list = []

        # Setup Environment
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        
        self.score_history = []
        self.learn_iters = 0
        self.avg_score = 0
        self.best_score = 0

        # Setup Agent
        self.agent = Agent(n_actions=self.action_space.n, batch_size=self.n_task, 
                           alpha=0.01, n_epochs=5, 
                           input_dims=self.observation_space.shape)

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
        support_losses = []
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

            support_losses.append(set_loss)
            
        avg_support_loss = torch.mean(torch.stack(support_losses))
        
        scores = self.forward(x_b_i)
        return avg_support_loss, scores

    def set_forward_loss(self, x):
        avg_support_loss, scores = self.set_forward(x, is_feature=False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        query_loss = self.loss_fn(scores, y_b_i)
        return avg_support_loss, query_loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_query_loss = 0
        task_count = 0
        all_query_loss = []

        # Reset environment
        score = 0
        done = False
        observation = np.zeros(2)

        optimizer.zero_grad()
        
        for i, (x, _) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"

            action, prob, val = self.agent.choose_action(observation)
            self.task_update_num = action + 1  # agent.step
            
            print('action:', self.task_update_num)
            print('prob:', prob)
            print('val:', val)

            avg_support_loss, query_loss = self.set_forward_loss(x)

            observation_ = np.array([avg_support_loss.item(), query_loss.item()], dtype=np.float32)
            print('observation: ', observation_)

            reward = np.array([-query_loss.item()])
            print('reward: ', reward)
            
            done = True if i == (len(train_loader) - 1) else False 
            score += reward
            self.agent.remember(observation, action, prob, val, reward, done)
            
            avg_query_loss += query_loss.item()
            all_query_loss.append(query_loss)

            task_count += 1
            if task_count == self.n_task:
                loss_q = torch.stack(all_query_loss).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()
                optimizer.step()

                task_count = 0
                all_query_loss = []

                self.agent.learn()
                self.learn_iters += 1
                
            optimizer.zero_grad()
            observation = observation_
            
            if i % print_freq == 0:
                print(f'Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss {avg_query_loss / float(i + 1):.6f}, Learn Iters {self.learn_iters}')

        self.score_history.append(score)
        self.avg_score = np.mean(self.score_history[-100:])

        if self.avg_score > self.best_score:
            self.best_score = self.avg_score
            self.agent.save_models()

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
