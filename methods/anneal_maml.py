# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
import math
import random

def half_trapezoidal_step_scheduler(total_epochs, current_epoch, max_step, min_step, max_step_width, half_right=True):

    max_step_start_epoch = math.floor(total_epochs - max_step_width*total_epochs) if half_right else math.floor(max_step_width*total_epochs)

    max_step_size = total_epochs - max_step_start_epoch if half_right else max_step_start_epoch

    step_freq = math.floor(max_step_start_epoch / (max_step - 1)) if half_right else math.floor((total_epochs - max_step_start_epoch) / (max_step-1))

    total_step_freq = (max_step - 1) * step_freq

    if total_step_freq < (total_epochs - max_step_size):
        step_diff = total_epochs - total_step_freq - max_step_size
        max_step_size += step_diff

    if half_right:
      # Determine the current step based on the current epoch
      if current_epoch < total_step_freq:
          current_step = min_step + current_epoch // step_freq
      else:
          current_step = max_step
    else:
      # Determine the current step based on the current epoch
      if current_epoch < max_step_size:
          current_step = max_step
      else:
          current_step = (max_step - 1) - (current_epoch - max_step_size) // step_freq

    return current_step
    
def new_trapezoidal_step_scheduler(total_epochs, current_epoch, max_step, min_step, max_step_width):
    step_diff = 0
    # Calculate the midpoint of total_epochs and max_step_width
    epoch_midpoint = total_epochs // 2
    max_step_width_midpoint = max_step_width / 2

    # Calculate the frequency of half max step and the starting and ending epochs of max step frequency
    half_max_step_freq = math.floor(total_epochs * max_step_width_midpoint)
    max_step_freq_start_epoch = epoch_midpoint - half_max_step_freq
    max_step_freq_end_epoch = epoch_midpoint + half_max_step_freq

    # Calculate the size of max step and total step frequency
    max_step_size = max_step_freq_end_epoch - max_step_freq_start_epoch
    total_step_freq_size = total_epochs - max_step_size

    # Calculate the size of half step frequency and the frequency of step
    half_step_freq_size = total_step_freq_size // 2
    step_freq = math.floor(half_step_freq_size / (max_step - 1))

    # Calculate the total step frequency and adjust max step size if necessary
    total_step_freq = (max_step - 1) * step_freq * 2
    if total_step_freq < (total_epochs - max_step_size):
        step_diff = total_epochs - total_step_freq - max_step_size
        max_step_size = max_step_size + step_diff if max_step_width != 0 else max_step_size

    # Calculate the new half step frequency size
    new_half_step_freq_size = total_step_freq // 2

    if max_step_width == 0 and step_diff > 0:
         # Determine the current step based on the current epoch
        if current_epoch < new_half_step_freq_size:
            current_step = min_step + current_epoch // step_freq
        elif current_epoch < (new_half_step_freq_size + step_diff):
            current_step = (max_step - 1)
        else:
            current_step = (max_step - 1) - (current_epoch - step_freq * (max_step - 1) - step_diff) // step_freq       
    else:
        # Determine the current step based on the current epoch
        if current_epoch < new_half_step_freq_size:
            current_step = min_step + current_epoch // step_freq
        elif current_epoch < (new_half_step_freq_size + max_step_size):
            current_step = max_step
        else:
            current_step = (max_step - 1) - (current_epoch - step_freq * (max_step - 1) - max_step_size) // step_freq

    return current_step

def trapezoidal_step_scheduler(total_epochs, current_epoch, max_adaptation_step, min_adaptation_step,  include_max_phase=True, half=False):
    # Guardrails to ensure correct types and values for the parameters
    assert isinstance(total_epochs, int), "total_epochs must be an integer"
    assert isinstance(current_epoch, int), "current_epoch must be an integer"
    assert isinstance(max_adaptation_step, int), "max_adaptation_step must be an integer"
    assert isinstance(min_adaptation_step, int), "min_adaptation_step must be an integer"
    assert isinstance(include_max_phase, bool), "include_max_phase must be a boolean"
    assert isinstance(half, bool), "half must be a boolean"
    
    assert min_adaptation_step > 0, "min_adaptation_step must be greater than 0"
    assert max_adaptation_step > min_adaptation_step, "max_adaptation_step must be greater than min_adaptation_step"

    # Check total_epochs based on the value of half and include_max_phase
    if not half:
      assert total_epochs >= 12 if include_max_phase else  total_epochs % max_adaptation_step == 0, "total_epochs must be at least 12 if include_max_phase, else total_epochs must be divisible by max_adaptation_step"

    # Determine the trapezoidal_region based on the value of half and include_max_phase
    trapezoidal_region = 2 if half or not include_max_phase else 3

    # Calculate the period and frequencies per step
    period = total_epochs // trapezoidal_region
    freq_per_step = math.floor(period / (max_adaptation_step - 1)) if include_max_phase else math.floor(period / (max_adaptation_step))
    freq_max_step = total_epochs - (trapezoidal_region - 1) * (freq_per_step * (max_adaptation_step - 1))

    # Determine the current step based on the current epoch and whether the max adaptation step phase is included
    if include_max_phase:
        if current_epoch < freq_per_step * (max_adaptation_step - 1):
            return min_adaptation_step + current_epoch // freq_per_step
        elif current_epoch < freq_per_step * (max_adaptation_step - 1) + freq_max_step:
            return max_adaptation_step
        else:
            return (max_adaptation_step - 1) - (current_epoch - freq_per_step * (max_adaptation_step - 1) - freq_max_step) // freq_per_step
    else:
        if current_epoch < freq_per_step * (max_adaptation_step):
            return min_adaptation_step + current_epoch // freq_per_step
        else:
            return max_adaptation_step - (current_epoch - freq_per_step * (max_adaptation_step)) // freq_per_step

class ANNEMAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, annealing_type = None, task_update_num_initial = None, task_update_num_final = None, annealing_rate = None, test_mode = False, approx = False):
        super(ANNEMAML, self).__init__( model_func,  n_way, n_support, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task     = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 0
        self.train_lr = 0.01 #this is the inner loop learning rate
        self.approx = approx #first order approx.    
        self.inner_loop_steps_list  = []  

        # annealing parameters
        print(f'Annealing params: {annealing_type}-{task_update_num_initial}-{task_update_num_final}-{annealing_rate}\n')
        self.annealing_type = annealing_type
        self.annealing_rate = annealing_rate  
        self.task_update_num_initial = task_update_num_initial
        self.task_update_num_final = task_update_num_final
        self.current_epoch = 0
        self.last_task_update_num = self.task_update_num_initial

        self.test_mode = test_mode
      
    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def annealing_func(self, task_update_num_final, task_update_num_initial, annealing_rate, current_epoch, atype=None):
      epochs = 200
      if atype == 'con':
        return task_update_num_initial
      # linear step
      elif atype == 'lin':
        return int(math.ceil(max(task_update_num_final, task_update_num_initial - annealing_rate * current_epoch))) 
      # inverse linear step    
      elif atype == 'lin_up':
        return int(math.floor(min(task_update_num_initial, task_update_num_final + annealing_rate * current_epoch)))
      # exp step
      elif atype == 'exp':
        return int(math.ceil(max(task_update_num_final, task_update_num_initial * np.exp(-annealing_rate * current_epoch))))
      # cosine step
      elif atype == 'cos':
        return int(math.ceil(max(task_update_num_final, task_update_num_initial * np.cos(annealing_rate * current_epoch))))
      # sigmoid step
      elif atype == 'sig':
        return int(math.ceil(max(task_update_num_final, task_update_num_initial / (1 + np.exp(annealing_rate * (current_epoch - epochs / 2))))))  
      # trapezoid step
      elif atype == 'tra':
        period = epochs // 3 
        if current_epoch < period:
          # Increase linearly
          return  int(math.ceil(task_update_num_final + (task_update_num_initial - task_update_num_final) * current_epoch / period))
        elif current_epoch < 2 * period:
            # Stay at maximum
            return int(task_update_num_initial)
        else:
            # Decrease linearly
            return int(math.ceil(task_update_num_initial - (task_update_num_initial - task_update_num_final) * (current_epoch - 2 * period) / period))
      # half trapezium step     
      elif atype == 'up_tra':
         period = epochs // 2 
         if current_epoch < period:
            # Increase linearly
            return  int(math.ceil(task_update_num_final + (task_update_num_initial - task_update_num_final) * current_epoch / period))
         else:
            # Stay at maximum
            return int(task_update_num_initial)
     # half trapezium step     
      elif atype == 'down_tra':
         period = epochs // 2 
         if current_epoch < period:
            # stay at maximum
            return int(task_update_num_initial)
         else:
            # Decrease linearly
            return int(math.ceil(max(task_update_num_final, task_update_num_initial - (task_update_num_initial - task_update_num_final) * (current_epoch - period) / period)))
     # triangle step     
      elif atype == 'tri':
         period = epochs // 2 
         if current_epoch < period:
            # Increase linearly
            return  int(math.ceil(task_update_num_final + (task_update_num_initial - task_update_num_final) * current_epoch / period))
         else:
            # Decrease linearly
            return int(math.ceil(max(task_update_num_final, task_update_num_initial - (task_update_num_initial - task_update_num_final) * (current_epoch - period) / period)))
      # random step
      elif atype == 'rand':
          # make sure the seed value is the same with the training script
          random.seed(10)
          # Generate a random number of steps within the range of initial and final
          return random.randint(task_update_num_final, task_update_num_initial)

      elif atype == 'tra_2':
          return trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_adaptation_step = task_update_num_initial, min_adaptation_step = task_update_num_final,  include_max_phase=True, half=False)

      elif atype == 'tri_2':
          return trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_adaptation_step = task_update_num_initial, min_adaptation_step = task_update_num_final,  include_max_phase=False, half=False)

      elif atype == 'up_tra_2':
          return trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_adaptation_step = task_update_num_initial, min_adaptation_step = task_update_num_final,  include_max_phase=True, half=True)

      elif atype == 'tra_3':
          return new_trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_step = task_update_num_initial, min_step = task_update_num_final, max_step_width = annealing_rate)

      elif atype == 'up_tra_3':
          return half_trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_step = task_update_num_initial, min_step = task_update_num_final, max_step_width = annealing_rate, half_right=True)
    
      elif atype == 'down_tra_3':
          return half_trapezoidal_step_scheduler(total_epochs = epochs, current_epoch = current_epoch, max_step = task_update_num_initial, min_step = task_update_num_final, max_step_width = annealing_rate, half_right=False)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'ANNEMAML do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        # do not anneal the inner steps in meta testing
        if self.test_mode:
            self.task_update_num = self.task_update_num_initial
        else:
            # Calculate task_update_num based on current epoch
            self.task_update_num = self.annealing_func(self.task_update_num_final, self.task_update_num_initial, self.annealing_rate, self.current_epoch, atype=self.annealing_type)

        # self.task_update_num = int(max(self.task_update_num_final, self.task_update_num_initial -  self.annealing_rate * self.current_epoch))

        # Print task_update_num if it has changed
        if self.task_update_num != int(self.last_task_update_num):
            print(f"task_update_num has changed to: {self.task_update_num}")
            self.last_task_update_num = self.task_update_num

        for task_step in range(self.task_update_num): 
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
            if self.approx:
                grad = [ g.detach()  for g in grad ] #do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts


        # feed forward query data
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('ANNEMAML performs further adapation simply by increasing task_upate_num')


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

        self.set_epoch(epoch)

        optimizer.zero_grad()

        #train
        for i, (x,_) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            

            loss = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
       
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()
                optimizer.step()
    
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                      
    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        # for i, (x,_) in enumerate(test_loader):
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
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
