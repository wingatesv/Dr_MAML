import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
from copy import deepcopy

class Reptile(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(Reptile, self).__init__( model_func,  n_way, n_support, change_way = False)
      
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task  = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.meta_step_size = 0.1


    def forward(self,x):
      out  = self.feature.forward(x)
      scores  = self.classifier.forward(out)
      return scores


    def set_forward(self,x, optimizer, is_feature = False):
        assert is_feature == False, 'Reptile do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_var = x_var.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:])  # Combine support and query sets
        y_var = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support + self.n_query))).cuda()  # Correctly generate labels
        
        

        for task_step in range(self.task_update_num): 
            scores = self.forward(x_var)

          
            loss = self.loss_fn( scores, y_var) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                  
            optimizer.zero_grad()

        return deepcopy(list(self.parameters()))  # Return the fine-tuned weights

                                

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')



    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        task_count = 0
        weights_original = deepcopy(self.state_dict())
        new_weights = []
      
        for i, (x,_) in enumerate(train_loader):
          
            adapted_weights = self.set_forward(x, optimizer)
            new_weights.append(adapted_weights)
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
                print('Epoch {:d} | Batch {:d}/{:d}'.format(epoch, i, len(train_loader)))

                        


                      
    
