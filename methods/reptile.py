# https://github.com/farbodtm/reptile-pytorch.git

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
from copy import deepcopy
import utils

class Reptile(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(Reptile, self).__init__( model_func,  n_way, n_support, change_way = False)
      
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        self.optimizer =  torch.optim.Adam(self.parameters(), lr = 0.0001, betas=(0, 0.999))
        
        self.n_task  = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.meta_step_size = 0.1
        self.test = False


    def forward(self,x):
      out  = self.feature.forward(x)
      scores  = self.classifier.forward(out)
      return scores


    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'Reptile do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data

        
      
        for task_step in range(self.task_update_num): 
            scores = self.forward(x_a_i)

          
            loss = self.loss_fn( scores, y_a_i) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                                  
            self.optimizer.zero_grad()
        
        if not self.test:
            return deepcopy(self.state_dict()), loss  

        else:
            scores = self.forward(x_b_i)
            return scores

                                

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')



    def train_loop(self, epoch, train_loader, optimizer=None): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        weights_original = deepcopy(self.state_dict())
        new_weights = []

        self.test = False
      
        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            adapted_weights, loss = self.set_forward(x)
            avg_loss = avg_loss+loss.item()
            new_weights.append(adapted_weights)
            # print(len(new_weights))
            self.load_state_dict(weights_original)
             
      
            task_count += 1

            if task_count == self.n_task:  # Meta update after processing n_task tasks
                ws = len(new_weights)
                fweights = {name: sum([new_weights[j][name] for j in range(ws)]) / float(ws) for name in new_weights[0]}

                self.load_state_dict({name: weights_original[name] + (fweights[name] - weights_original[name]) * self.meta_step_size
                                      for name in weights_original})

                task_count = 0  # Reset task count
                new_weights = []  # Clear new weights list

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        avg_loss=0
        self.test = True
        acc_all = []
        weights_original = deepcopy(self.state_dict())
        
        iter_num = len(test_loader) 
        # for i, (x,_) in enumerate(test_loader):
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "Reptile do not support way change"
            correct_this, count_this, loss = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )
            avg_loss = avg_loss+loss.item()

            self.load_state_dict({ name: weights_original[name] for name in weights_original })
            

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% ± %4.2f%%, Test Loss = %4.4f' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num), float(avg_loss/iter_num)))
        if return_std:
            return acc_mean, acc_std, float(avg_loss/iter_num)
        else:
            return acc_mean, float(avg_loss/iter_num)
