import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
import utils

class Regularizer(nn.Module):
    def __init__(self, input_dim=36):
        super(Regularizer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 36),      # Input layer
            nn.ReLU(inplace=True),
            nn.Linear(36, 36)        
        )
    
    def forward(self, x):
        return self.network(x)

class MAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, approx=False, alfa=True):
        super(MAML, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task = 4  # meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.train_lr = 0.01  # this is the inner loop learning rate
        self.approx = approx  # first order approx.
        self.alfa = alfa  # enable ALFA mechanism

        # Initial learning rate and weight decay
        self.init_learning_rate = 1e-3  # from the repo's implementation
        self.init_weight_decay = 5e-4   # from the repo's implementation
        # Print number of parameters before initializing post-multipliers
        num_params_before = sum(p.numel() for p in self.parameters())
        print(f'Number of parameters before initializing post-multipliers: {num_params_before}')

        # Meta-learnable post-multipliers
        if self.alfa:
            # Convert the generator to a list to get the number of parameters
            parameter_list = list(self.parameters())
            self.alpha_post_multipliers = nn.ParameterList([
                nn.Parameter(torch.ones(self.task_update_num) * self.init_learning_rate) for _ in range(len(parameter_list))
            ])
            self.beta_post_multipliers = nn.ParameterList([
                nn.Parameter(torch.ones(self.task_update_num) * self.init_weight_decay * self.init_learning_rate) for _ in range(len(parameter_list))
            ])
        # Print number of parameters after initializing post-multipliers
        num_params_after = sum(p.numel() for p in self.parameters())
        print(f'Number of parameters after initializing post-multipliers: {num_params_after}')

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def set_forward(self, x, regularizer=None, is_feature=False):
        assert not is_feature, 'MAML does not support fixed feature'
        # regularizer = regularizer

        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:, :self.n_support, :, :, :].contiguous().view(self.n_way * self.n_support, *x.size()[2:])  # support data
        x_b_i = x_var[:, self.n_support:, :, :, :].contiguous().view(self.n_way * self.n_query, *x.size()[2:])  # query data
        y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data

        # fast_parameters = list(self.parameters())  # the first gradient calculated is based on original weight
        fast_parameters = [param for name, param in self.named_parameters() if "alpha_post_multipliers" not in name and "beta_post_multipliers" not in name]
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

       

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)  # build full graph support gradient of gradient

            if self.approx:
                grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation

            if self.alfa and regularizer is not None:
               # Generate layer-wise means of gradients and weights
                per_step_task_embedding = []
                # print(f"Task Step {task_step}:")
                for k, weight in enumerate(fast_parameters):
                    weight_mean = weight.mean()
                    grad_mean = grad[k].mean()
                     # Print the means for debugging
                    # print(f"Layer {k} - Weight Mean: {weight_mean.item()}, Grad Mean: {grad_mean.item()}")

                    per_step_task_embedding.append(weight_mean)
                    per_step_task_embedding.append(grad_mean)

                per_step_task_embedding = torch.stack(per_step_task_embedding)
                # print(f"per_step_task_embedding shape: {per_step_task_embedding.shape}")

                generated_params = regularizer(per_step_task_embedding)
                # print('shape of output: ', generated_params.shape)

                num_layers = len(fast_parameters)
                generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=num_layers)
                # print('alpha shape', generated_alpha.shape)
                # print('beta shape', generated_beta.shape)

            # Assuming generated_alpha and generated_beta are of correct length
            num_layers = len(fast_parameters)  # This should match the length of post-multipliers

            # Ensure the lengths match
            assert num_layers == len(self.alpha_post_multipliers), "Mismatch in number of layers and alpha post-multipliers"
            assert num_layers == len(self.beta_post_multipliers), "Mismatch in number of layers and beta post-multipliers"
            # Get the relevant parameters for the model layers, excluding alpha and beta post-multipliers
            relevant_parameters = [
                param for name, param in self.named_parameters()
                if "alpha_post_multipliers" not in name and "beta_post_multipliers" not in name
            ]
            fast_parameters = []
            for k, weight in enumerate(relevant_parameters):
                if weight.fast is None:
                    if self.alfa and regularizer is not None:
                        # print(f"k={k}, task_step={task_step}, num_layers={num_layers}")
                        alpha = generated_alpha[k] * self.alpha_post_multipliers[k][task_step]
                        beta = generated_beta[k] * self.beta_post_multipliers[k][task_step]
                        weight.fast = weight - alpha * grad[k] - beta
                    else:
                        weight.fast = weight - self.train_lr * grad[k]  # create weight.fast
                else:
                    if self.alfa and regularizer is not None:
                        alpha = generated_alpha[k] * self.alpha_post_multipliers[k][task_step]
                        beta = generated_beta[k] * self.beta_post_multipliers[k][task_step]
                        weight.fast = weight.fast - alpha * grad[k] - beta
                    else:
                        weight.fast = weight.fast - self.train_lr * grad[k]  # create an updated weight.fast

                fast_parameters.append(weight.fast)

        # feed forward query data
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self, x, is_feature=False):  # overwrite parent function
        raise ValueError('MAML performs further adaptation simply by increasing task_update_num')

    def set_forward_loss(self, x, regularizer=None):
        scores = self.set_forward(x, regularizer=regularizer, is_feature=False)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss

    def train_loop(self, epoch, train_loader, optimizer, regularizer=None):  # overwrite parent function
        print_freq = 10
        avg_loss = 0
        task_count = 0
        loss_all = []

        optimizer.zero_grad()

        # train
        for i, (x, _) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"

            loss = self.set_forward_loss(x, regularizer=regularizer)
            avg_loss += loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task:  # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()
                optimizer.step()

                task_count = 0
                loss_all = []
            optimizer.zero_grad()

            if i % print_freq == 0:
                print(f'Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss {avg_loss / float(i + 1):.6f}')

    def correct(self, x, regularizer):       
        scores = self.set_forward(x, regularizer=regularizer)
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        
        if hasattr(self, 'loss_type') and self.loss_type == 'mse':
            y = utils.one_hot(y, self.n_way)
            
        y = Variable(y.cuda())
        loss = self.loss_fn(scores, y)

        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query), loss

    def test_loop(self, test_loader, regularizer=None, return_std=False):  # overwrite parent function
        correct = 0
        count = 0
        avg_loss = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0), "MAML does not support way change"
            correct_this, count_this, loss = self.correct(x, regularizer)
            acc_all.append(correct_this / count_this * 100)
            avg_loss += loss.item()

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(f'{iter_num} Test Acc = {acc_mean:.2f}% ± {1.96 * acc_std / np.sqrt(iter_num):.2f}%, Test Loss = {avg_loss / iter_num:.4f}')
        if return_std:
            return acc_mean, acc_std, avg_loss / iter_num
        else:
            return acc_mean, avg_loss / iter_num
