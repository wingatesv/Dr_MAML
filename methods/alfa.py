# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 
import os
import torch.optim 
import backbone
import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm

# Define the hyperparameter generator network
class HyperparameterGenerator(nn.Module):
    def __init__(self):
        super(HyperparameterGenerator, self).__init__()
        # Fixed dimensions for the hidden layers and output
        self.fc1_dims = 32
        self.fc2_dims = 32
        self.output_dims = 2


        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(1, self.fc1_dims),  # Placeholder for input dimensions
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.output_dims),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, gradients, weights):
        # Compute the layer-wise means
        device = next(self.parameters()).device
        grad_means = [torch.mean(g) for g in gradients]
        weight_means = [torch.mean(w) for w in weights]
        
        # Concatenate and pass through the network to get alpha and beta
        inputs = torch.cat([torch.stack(grad_means), torch.stack(weight_means)], dim=0).unsqueeze(0)
        
        # Update input dimension based on the concatenated tensor
        self.network[0] = nn.Linear(inputs.size(1), self.fc1_dims).to(device)
        
        alpha, beta = self.network(inputs).squeeze(0)
        return alpha, beta



class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(MAML, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task     = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.approx = approx #first order approx.    
        self.inner_loop_steps_list  = []  

       
        # self.hyperparameter_net = HyperparameterGenerator()
        self.train_lr = 0.01 #this is the inner loop learning rate
        self.reg_lambda = 0.01

        #Create tensors for train_lr and reg_lambda with the same size as the model parameters
        # self.train_lr = nn.ParameterList([nn.Parameter(torch.ones_like(param) * 0.01) for param in self.parameters()])
        # self.reg_lambda = nn.ParameterList([nn.Parameter(torch.ones_like(param) * 0.01) for param in self.parameters()])



    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self,x, hyperparameter_net, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()


        for task_step in range(self.task_update_num): 

  
            
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation

            # Extract gradients and weights for the hyperparameter generator
            gradients = grad
            weights = list(self.parameters())

            # Use the hyperparameter network to get updated learning rate and regularization
            train_lr, reg_lambda = hyperparameter_net(gradients, weights)
            print('Train_lr:', train_lr.item())
            print('Reg_lambda:', reg_lambda.item())


            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * (grad[k] + self.reg_lambda * weight) #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * (grad[k] + self.reg_lambda * weight) #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts


        # feed forward query data
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x, hyperparameter_net):
        scores = self.set_forward(x, hyperparameter_net, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)

        return loss


    def train_loop(self, epoch, train_loader, optimizer, hyperparameter_net): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []


        optimizer.zero_grad()
        hyperparameter_net.optimizer.zero_grad()

        #train
        for i, (x,_) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            

            loss = self.set_forward_loss(x, hyperparameter_net)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)


            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
       
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()
                optimizer.step()


                hyperparameter_net.optimizer.step()
    
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            hyperparameter_net.optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

    def correct(self, x, hyperparameter_net):       
        scores = self.set_forward(x, hyperparameter_net)
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
                      
    def test_loop(self, test_loader, hyperparameter_net, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        avg_loss=0
        acc_all = []
        
        iter_num = len(test_loader) 
        # for i, (x,_) in enumerate(test_loader):
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this, loss = self.correct(x, hyperparameter_net)
            acc_all.append(correct_this/ count_this *100 )
            avg_loss = avg_loss+loss.item()

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% ± %4.2f%%, Test Loss = %4.4f' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num), float(avg_loss/iter_num)))
        if return_std:
            return acc_mean, acc_std, float(avg_loss/iter_num)
        else:
            return acc_mean, float(avg_loss/iter_num)



