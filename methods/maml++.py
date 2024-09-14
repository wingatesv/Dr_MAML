# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class MAML_plus_plus(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, test_mode = False, total_epochs=200, approx_until_epoch=50):
        super(MAML_plus_plus, self).__init__( model_func,  n_way, n_support, change_way = False)
        self.task_update_num = 5
        # Initialize ConvNet_bnwb
        self.feature = backbone.ConvNet_bnwb(
            depth=4,
            flatten=True,
            num_steps=self.task_update_num + 1,
            maml=True
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.feat_dim = self.feature.final_feat_dim
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        self.test_mode  = test_mode
        self.n_task     = 4 #meta-batch, meta update every meta batch
        
        self.train_lr = 0.01 #this is the inner loop learning rate
        self.approx = True #first order approx.    
        self.approx_until_epoch = approx_until_epoch
        self.current_epoch = 0
        self.total_epochs = total_epochs

    
        self.grad_norm = 0
        self.train_confidence = 0
        self.train_entropy = 0
        self.train_loss = 0
        self.val_loss = 0

        
         # Store metrics for plotting
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'grad_norm': [],
            'acc_mean': [],
            'confidence': [],
            'entropy': [],
        }
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        # Update approx flag based on current epoch
        if self.current_epoch >= self.approx_until_epoch:
            self.approx = False  # Switch to second-order gradients
        else:
            self.approx = True

    def collect_metrics(self, acc_mean):
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['train_loss'].append(self.train_loss)
        self.metrics['val_loss'].append(self.val_loss)
        self.metrics['grad_norm'].append(self.grad_norm)
        self.metrics['confidence'].append(self.train_confidence)
        self.metrics['entropy'].append(self.train_entropy)
        self.metrics['acc_mean'].append(acc_mean/100)
        print(f"Metrics collected for epoch {self.current_epoch}.")


    def forward(self,x, step=None):
        out  = self.feature.forward(x, step=step)
        scores  = self.classifier.forward(out)
        return scores
    
    def compute_loss_weights(self):
        t = self.current_epoch / self.total_epochs  # Normalized epoch (0 to 1)
        # Start with equal weights
        weights = np.ones(self.task_update_num) / self.task_update_num
        # Define weights focusing more on later steps as training progresses
        weights_last_step = np.zeros(self.task_update_num)
        weights_last_step[-1] = 1.0
        # Linearly interpolate between equal weights and final step focus
        weights = (1 - t) * weights + t * weights_last_step
        return weights
        
    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        total_loss = 0
        v_i = self.compute_loss_weights()
        v_i = torch.tensor(v_i, dtype=torch.float32).cuda()

        for task_step in range(self.task_update_num): 
            scores = self.forward(x_a_i, step=task_step)
            set_loss = self.loss_fn( scores, y_a_i) 
            create_graph_flag = not self.approx
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=create_graph_flag)

            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k] #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast 
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts


            # feed forward query data
            scores_query = self.forward(x_b_i, step=task_step+1)
            query_loss = self.loss_fn(scores_query, y_b_i)
            total_loss += v_i[task_step] * query_loss

        if not self.test_mode:
            return total_loss, scores_query
        else:
            return scores_query
  

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        total_loss, scores_query = self.set_forward(x, is_feature = False)
        return total_loss, scores_query


    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        self.test_mode = False
        self.set_epoch(epoch)
        optimizer.zero_grad()

        all_confidences = []
        all_entropies = []
        grad_norms = []

        #train
        for i, (x,_) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            

            loss, scores = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            # Compute softmax probabilities
            probs = F.softmax(scores, dim=1)
            
            # Compute confidence as max probability for each query
            confidence = probs.max(dim=1)[0].mean().item()
            all_confidences.append(confidence)
            
            # Compute entropy for each query
            epsilon = 1e-8
            probs = torch.clamp(probs, min=epsilon, max=1.0)
            entropy = -(probs * probs.log()).sum(dim=1).mean().item()
            all_entropies.append(entropy)

            task_count += 1

            if task_count == self.n_task: #MAML update several tasks at one time
       
                loss_q = torch.stack(loss_all).sum(0)
                loss_value = loss_q.item()
                loss_q.backward()

                grad_norm = 0
                # Calculate gradient norm
                for param in self.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5  # Take the square root to get the norm
                grad_norms.append(grad_norm)

                optimizer.step()
    
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
        self.train_loss = avg_loss/len(train_loader)
        self.train_confidence = sum(all_confidences) / len(all_confidences)
        self.train_entropy = sum(all_entropies) / len(all_entropies)
        self.grad_norm = sum(grad_norms) / len(grad_norms)

    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
        correct =0
        count = 0
        avg_loss=0
        self.test_mode = True
        acc_all = []
        
        iter_num = len(test_loader) 
        # for i, (x,_) in enumerate(test_loader):
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this, loss = self.correct(x)
            acc_all.append(correct_this/ count_this *100 )
            avg_loss = avg_loss+loss.item()

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)

        if not self.test_mode:
          self.val_loss = avg_loss/len(test_loader)
          self.collect_metrics(acc_mean)

        print('%d Test Acc = %4.2f%% ± %4.2f%%, Test Loss = %4.4f' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num), float(avg_loss/iter_num)))
        if return_std:
            return acc_mean, acc_std, float(avg_loss/iter_num)
        else:
            return acc_mean, float(avg_loss/iter_num)


    # New method to output metrics and plot graphs
    def output_metrics(self, file_path=None, save_plots=True, plot_dir='/content/'):
        # Convert the metrics dictionary to a pandas DataFrame
        metrics_df = pd.DataFrame(self.metrics)
        
        if file_path:
            # Save the DataFrame as a CSV file
            metrics_df.to_csv(file_path, index=False)
            print(f"Metrics saved to {file_path}")
            
       
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
            'confidence': '#1f77b4',  # Light blue
             'entropy': '#9467bd',    # Light green
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

        # Second plot: confidence, entropy and Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epochs'], metrics_df['confidence'], label='Train Confidence', color=color_scheme['confidence'])
        plt.plot(metrics_df['epochs'], metrics_df['entropy'], label='Train Entropy', color=color_scheme['entropy'])
        plt.plot(metrics_df['epochs'], metrics_df['acc_mean'], label='Accuracy', color=color_scheme['acc_mean'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend(loc='best')
        if save_plots:
            plt.savefig(f'{plot_dir}train_confidence_entropy_acc_plot.png', bbox_inches='tight')
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
        

        # Return the DataFrame for further use if needed
        return metrics_df
