# This code is modified from https://github.com/dragen1860/Aux_MAML-Pytorch and https://github.com/katerakelly/pytorch-Aux_MAML 

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
import random
import cv2



class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super(StainNet, self).__init__()
        model_list = []
        model_list.append(nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
        model_list.append(nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
            model_list.append(nn.ReLU(True))
        model_list.append(nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))

        self.rgb_trans = nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)

class Aux_MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False, test_mode = False):
        super(Aux_MAML, self).__init__( model_func,  n_way, n_support, change_way = False)
        self.aux_task = 'sn' #inpainting, segmentation
        print('aux_task:', self.aux_task)
        self.feature = backbone.ConvNet(4, flatten=False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        self.test_mode  = test_mode
        self.n_task     = 4 #meta-batch, meta update every meta batch
        self.task_update_num = 5
        self.train_lr = 0.01 #this is the inner loop learning rate
        self.approx = approx #first order approx.    
        self.inner_loop_steps_list  = []  
        self.grad_norm = 0
        self.train_confidence = 0
        self.train_entropy = 0
        self.train_loss = 0
        self.val_loss = 0
        self.current_epoch = 0

        if self.aux_task == 'sn':
            # Initialize the StainNet model
            STAINNET_WEIGHTS = '/content/Dr_MAML/data/StainNet-Public-centerUni_layer3_ch32.pth'
            self.stainnet_model = StainNet().cuda()
            self.stainnet_model.load_state_dict(torch.load(STAINNET_WEIGHTS, weights_only=True))
            self.stainnet_model.eval()  # Set the model to evaluation mode

        output_channels = 3 if self.aux_task in ['inpainting', 'sn'] else 1
        print(output_channels)
        self.inpainting_head = backbone.InpaintingHead(input_channels=64, output_channels=output_channels)
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
    def parameters(self):
        # Override the parameters method to exclude StainNet's parameters
        return (param for name, param in self.named_parameters() if not name.startswith('stainnet_model'))
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def collect_metrics(self, acc_mean):
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['train_loss'].append(self.train_loss)
        self.metrics['val_loss'].append(self.val_loss)
        self.metrics['grad_norm'].append(self.grad_norm)
        self.metrics['confidence'].append(self.train_confidence)
        self.metrics['entropy'].append(self.train_entropy)
        self.metrics['acc_mean'].append(acc_mean/100)
        print(f"Metrics collected for epoch {self.current_epoch}.")
      
    def forward(self,x):
        out  = self.feature.forward(x)
        out = out.view(out.size(0), -1)
        scores  = self.classifier.forward(out)
        return scores

    def unnormalize(self, images, mean, std):
        # images: Tensor of shape (batch_size, channels, H, W)
        # mean and std: lists of mean and std values for each channel
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
        images = images * std + mean
        return images

    def normalize(self, images, mean, std):
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)
        images = (images - mean) / std
        return images

    def stain_normalize(self, images):
      # images: Tensor of shape (batch_size, 3, H, W), expected in range [0, 1] or [-1, 1]
  
      # Clone images to avoid modifying the original tensor
      images = images.clone()

      # Define mean and std used during normalization
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
    
      # Unnormalize the images to bring them back to [0, 1] range
      images = self.unnormalize(images, mean, std)


      # Check the pixel value range
      # min_val = images.min()
      # max_val = images.max()
      # print(f'Input images pixel range: [{min_val.item()}, {max_val.item()}]')
  
      # Ensure images are in the range [-1, 1]
      if images.min() >= 0 and images.max() <= 1:
          images = images * 2 - 1  # Scale from [0, 1] to [-1, 1]
      elif images.min() >= -1 and images.max() <= 1:
          pass  # Images are already in [-1, 1]
      else:
          raise ValueError('Input images should have pixel values in the range [0, 1] or [-1, 1]')
  
      # Run the images through the StainNet model
      self.stainnet_model.eval()  # Ensure the model is in evaluation mode
      with torch.no_grad():
          normalized_images = self.stainnet_model(images)
  
      # The output of StainNet is in the range [-1, 1]; convert back to [0, 1]
      normalized_images = (normalized_images + 1) / 2

      # Re-apply the normalization
      normalized_images = self.normalize(normalized_images, mean, std)
  
      # Ensure the output has the same shape as the input
      assert normalized_images.shape == images.shape, \
          f'Expected output shape {images.shape}, but got {normalized_images.shape}'
  
      return normalized_images
        
    def random_block_mask(self, images, mask_size_ratio=0.1):
        batch_size, _, h, w = images.shape
        masks = torch.ones((batch_size, 1, h, w), device=images.device)
        block_size = int((mask_size_ratio ** 0.5) * h)  # Block side based on the desired mask ratio
    
        for i in range(batch_size):
            y = random.randint(0, h - block_size)
            x = random.randint(0, w - block_size)
            masks[i, :, y:y + block_size, x:x + block_size] = 0  # Mask out a block in each image
    
        masks = masks.expand(-1, images.size(1), -1, -1)  # Expand to match the number of channels
        masked_images = images * masks
        return masked_images, masks

    def generate_otsu_mask(self, image_batch):
        batch_size, _, h, w = image_batch.size()
        masks = torch.zeros(batch_size, 1, h, w).cuda()
    
        for i in range(batch_size):
            # Convert to numpy, grayscale, and apply Otsu's thresholding
            image_np = image_batch[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian Blur
            gray_image_uint8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
            # Apply Otsu's thresholding
            _, mask = cv2.threshold(gray_image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to tensor and normalize mask values to 0 and 1
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            masks[i] = mask
    
        return masks.cuda()
        
    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'Aux_MAML do not support fixed feature' 
        
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) #support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) #query data
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() #label for support data


        if self.aux_task == 'sn':
            # Generate stain-normalized images for the support data
            with torch.no_grad():
                stain_normalized_images = self.stain_normalize(x_a_i)  # Function to generate stain-normalized images
        elif self.aux_task == 'inpainting':
            # Generate masked images and masks for the inpainting task
            masked_images, masks = self.random_block_mask(x_a_i)
        elif self.aux_task == 'segmentation':
            #  Generate segmentation masks using Otsu's method
            tissue_masks = self.generate_otsu_mask(x_a_i)  # Generates binary masks for support data
        
        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()


        for task_step in range(self.task_update_num): 
            scores = self.forward(x_a_i)
            set_loss_cls = self.loss_fn( scores, y_a_i) 

            if self.aux_task == 'sn':
                # Inpainting forward pass
                features = self.feature(x_a_i)
                predicted_images = self.inpainting_head(features)
                # Compute stain normalization loss
                aux_loss = nn.L1Loss()(predicted_images, stain_normalized_images)

            elif self.aux_task == 'inpainting':
                 # Inpainting forward pass
                features = self.feature(masked_images)
                reconstructed_images = self.inpainting_head(features)
                # Compute inpainting loss on masked regions
                aux_loss = F.mse_loss(reconstructed_images * (1 - masks), x_a_i * (1 - masks))
                
            elif self.aux_task == 'segmentation':
                 # Inpainting forward pass
                features = self.feature(x_a_i)
                predicted_masks = self.inpainting_head(features)
                # Compute inpainting loss on masked regions
                aux_loss = nn.BCELoss()(predicted_masks, tissue_masks)  # Binary cross-entropy loss
            
    
            # Total loss
            total_loss = set_loss_cls + 0.5 * aux_loss

            grad = torch.autograd.grad(total_loss, fast_parameters, create_graph=True) #build full graph support gradient of gradient
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
        raise ValueError('Aux_MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)

        return loss, scores


    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        self.set_epoch(epoch)
        optimizer.zero_grad()

        all_confidences = []
        all_entropies = []
        grad_norms = []

        #train
        for i, (x,_) in enumerate(train_loader):

            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "Aux_MAML do not support way change"
            

            loss, scores = self.set_forward_loss(x)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            # Compute softmax probabilities
            probs = F.softmax(scores, dim=1)
            
            # Compute confidence as max probability for each query
            confidence = probs.max(dim=1)[0].mean().item()
            all_confidences.append(confidence)
            
            # Compute entropy for each query
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
            all_entropies.append(entropy)

            task_count += 1

            if task_count == self.n_task: #Aux_MAML update several tasks at one time
       
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
        acc_all = []
        
        iter_num = len(test_loader) 
        # for i, (x,_) in enumerate(test_loader):
        for i, (x,_) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "Aux_MAML do not support way change"
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

        # Second plot: Confidnece, ENtropy and Accuracy
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