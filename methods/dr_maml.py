# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict

from losses import SupConLoss

class DR_MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = True):
        super(DR_MAML, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.supcon_loss = SupConLoss(temperature=0.07,)
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task     = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.approx = approx #first order approx.    
        self.task_lr = OrderedDict()

        
    def define_task_lr_params(self):
         for k, param in enumerate(self.classifier.parameters()): 
              self.task_lr[k] =  nn.Parameter(1e-2 * torch.ones_like(param, requires_grad=True))


    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return out, scores

    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'DR_MAML do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        fast_parameters = list(self.classifier.parameters())
        for weight in self.classifier.parameters():
            weight.fast = None
        self.zero_grad()


        for task_step in range(self.task_update_num): 
            out, scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation

            fast_parameters = []
            for k, weight in enumerate(self.classifier.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.task_lr[k] * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.task_lr[k] * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
            

        # feed forward query data
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('DR_MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)

        return loss


    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []

        optimizer.zero_grad()

        #train
        for i, (x,_) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "ANIL do not support way change"
            
            # self.approx = False if epoch < 50 else True
            loss = self.set_forward_loss(x)

            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)



            task_count += 1


            if task_count == self.n_task: #MAML update several tasks at one time
       
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()


                loss_q.backward()

                # Check if gradients have been computed for each parameter
                # for name, param in self.named_parameters():
                #     if param.grad is not None:
                #         if name  in self.task_lr:
                #           print(f"Gradients computed for parameter '{name}'")
                #           # # print('grad: ',param.grad.shape)
                #           # lr_param_name = f"learning_rates.{name}"
                #           # print(lr_param_name)
                          
                #           lr_param = self.task_lr[name]
                #           print('before:', lr_param)
                #           print(lr_param.shape)
                #           lr_grad = param.grad.clone()
                #           lr_param -= 0.001 * lr_grad
                #           print('after:', lr_param)
                #     else:
                #         print(f"No gradients computed for parameter '{name}'")
                  

                  

                # for k, (name, lr) in enumerate(self.learning_rates.items()):
                # # Update the learnable learning rate based on your desired logic
                #   print(lr.grad)
                #   lr.data -= 0.01
                #   print(f"Learning rate for {name}: {lr.item()}")

                optimizer.step()
                # print('Check value: ',self.task_lr)
                task_count = 0
                loss_all = []

          
            optimizer.zero_grad()
            
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

               # # Update the learning rate
                # if scheduler != None:
                #   scheduler.step()
                #   print('Learning rate: ',optimizer.param_groups[0]['lr'])

                      
    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "IMAML_IDCG do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% ± %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
