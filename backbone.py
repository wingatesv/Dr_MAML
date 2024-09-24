# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
# from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.utils.parametrizations import weight_norm as WeightNorm

import torchvision.models as models
import torch.nn.init as init



def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            # WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      
            self.L = WeightNorm(self.L, name='weight', dim=0)
        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out



class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out


class ResNet_ImageNet(nn.Module):
  def __init__(self, flatten = True, model = 'resnet18', frozen = True, layers=6):

    super(ResNet_ImageNet, self).__init__()
    self.name = model
    self.flatten = flatten
    last_layer = -1 if flatten else -2

    if self.name =='resnet18':
      self.weights = models.ResNet18_Weights.IMAGENET1K_V1
      self.final_feat_dim = 512 if flatten else [512, 7, 7] #self.feature_extractor.fc.in_features
    elif self.name == 'resnet34':
      self.weights = models.ResNet34_Weights.IMAGENET1K_V1
      self.final_feat_dim = 512 if flatten else [512, 7, 7]
    elif self.name == 'resnet50':
      self.weights = models.ResNet50_Weights.IMAGENET1K_V1
      self.final_feat_dim = 2048 if flatten else [2048, 7, 7]
    elif self.name == 'resnet101':
      self.weights = models.ResNet101_Weights.IMAGENET1K_V1
      self.final_feat_dim = 2048 if flatten else [2048, 7, 7]


    self.feature_extractor = models.__dict__[self.name](weights=self.weights)

    self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:last_layer])

    #  set backbone training to False
    if frozen:
      print(f'Using Frozen ImageNet pre-trained weights, with last {layers} unfrozen')
      for param in self.feature_extractor.parameters():
        param.requires_grad = False
      self.unfreeze_layers(num_layers = layers)
      self.check_requires_grad()
      
    
    # if frozen:
    #   print('Freezing ImageNet pre-trained weights in Block 1 to 6 only, Block 7 (Last block) requires grad: True')
    #   for name, module in self.feature_extractor.named_children():
    #       if name != '7':
    #         for param in module.parameters():
    #           param.requires_grad = False
    #   self.check_requires_grad()


  def unfreeze_layers(self, num_layers):
        total_layers = len(list(self.feature_extractor.parameters()))
        layers_to_unfreeze = total_layers - num_layers
        count = 0

        for module in self.feature_extractor.children():
            for param in module.parameters():
                if count >= layers_to_unfreeze:
                    param.requires_grad = True
                count += 1


  def forward(self, x):
    x = self.feature_extractor(x)
    if self.flatten:
      x = nn.functional.adaptive_avg_pool2d(x, output_size=(1,1))
      x = x.view(x.size(0), -1)
    return x
  
  def check_requires_grad(self):
    for name, param in self.named_parameters():
      if param.requires_grad:
         print(f'Layer: {name}\tRequires Grad: {param.requires_grad}')



class BatchNorm2d_fw_bnrs(nn.BatchNorm2d):
    def __init__(self, num_features, num_steps):
        super(BatchNorm2d_fw_bnrs, self).__init__(num_features, affine=True, track_running_stats=False)
        self.weight.fast = None
        self.bias.fast = None

        # Per-step running statistics
        self.num_steps = num_steps
        self.running_means = [torch.zeros(num_features).cuda() for _ in range(num_steps)]
        self.running_vars = [torch.ones(num_features).cuda() for _ in range(num_steps)]

    def forward(self, x, step):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
    
        batch_mean = x.mean([0, 2, 3])
        batch_var = x.var([0, 2, 3], unbiased=False)
        momentum = 0.1
    
        # Update running statistics
        running_mean = (1 - momentum) * self.running_means[step] + momentum * batch_mean.data
        running_var = (1 - momentum) * self.running_vars[step] + momentum * batch_var.data
    
        self.running_means[step] = running_mean
        self.running_vars[step] = running_var
    
        out = F.batch_norm(
            x, self.running_means[step], self.running_vars[step], weight, bias,
            training=False, momentum=0.1, eps=self.eps
        )
    
        return out

class BatchNorm2d_fw_bnwb(nn.BatchNorm2d):
    def __init__(self, num_features, num_steps):
        super(BatchNorm2d_fw_bnwb, self).__init__(num_features, affine=False, track_running_stats=False)
        self.num_features = num_features
        self.num_steps = num_steps

        # Initialize per-step weights and biases
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(num_features)) for _ in range(num_steps)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(num_features)) for _ in range(num_steps)])

        # Fast weights for inner-loop updates
        self.weights_fast = [None] * num_steps
        self.biases_fast = [None] * num_steps

        # Initialize per-step running statistics
        self.running_means = [torch.zeros(num_features) for _ in range(num_steps)]
        self.running_vars = [torch.ones(num_features) for _ in range(num_steps)]

    def forward(self, x, step):
        if self.weights_fast[step] is not None and self.biases_fast[step] is not None:
            weight = self.weights_fast[step]
            bias = self.biases_fast[step]
        else:
            weight = self.weights[step]
            bias = self.biases[step]


        running_mean = self.running_means[step].to(x.device)
        running_var = self.running_vars[step].to(x.device)

        # Compute batch statistics
        batch_mean = x.mean([0, 2, 3])
        batch_var = x.var([0, 2, 3], unbiased=False)
        momentum = 0.1

        # Update running statistics
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean.data
        running_var = (1 - momentum) * running_var + momentum * batch_var.data

        self.running_means[step] = running_mean
        self.running_vars[step] = running_var

        # Normalize
        out = F.batch_norm(
            x, running_mean, running_var, weight, bias,
            training=False, momentum=0.0, eps=self.eps
        )

        return out


class ConvBlock_bnrs(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1, num_steps=6, maml=True):
        super(ConvBlock_bnrs, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.maml = maml

        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw_bnrs(outdim, num_steps=num_steps)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x, step=None):
        out = x
        for layer in self.parametrized_layers:
            if isinstance(layer, BatchNorm2d_fw_bnrs):
                out = layer(out, step)
            else:
                out = layer(out)
        return out

class ConvBlock_bnwb(nn.Module):
    def __init__(self, indim, outdim, pool=True, padding=1, num_steps=6, maml=True):
        super(ConvBlock_bnwb, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.maml = maml

        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw_bnwb(outdim, num_steps=num_steps)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x, step=None):
        out = x
        for layer in self.parametrized_layers:
            if isinstance(layer, BatchNorm2d_fw_bnwb):
                out = layer(out, step)
            else:
                out = layer(out)
        return out


class ConvNet_bnrs(nn.Module):
    def __init__(self, depth, flatten=True, num_steps=6, maml=True):
        super(ConvNet_bnrs, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock_bnrs(indim, outdim, pool=(i < 4), num_steps=num_steps, maml=maml)
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x, step=None):
        out = x
        for module in self.trunk:
            if isinstance(module, ConvBlock_bnrs):
                out = module(out, step)
            else:
                out = module(out)
        return out

class ConvNet_bnwb(nn.Module):
    def __init__(self, depth, flatten=True, num_steps=6, maml=True):
        super(ConvNet_bnwb, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock_bnwb(indim, outdim, pool=(i < 4), num_steps=num_steps, maml=maml)
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x, step=None):
        out = x
        for module in self.trunk:
            if isinstance(module, ConvBlock_bnwb):
                out = module(out, step)
            else:
                out = module(out)
        return out


class InpaintingHead(nn.Module):
    def __init__(self, input_channels=64, output_channels=3):
        super(InpaintingHead, self).__init__()
        
        # Upsampling layers (Upsample + Conv2d)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.01, inplace=True)
        )
        # Final convolution
        self.conv_final = nn.Conv2d(8, output_channels, kernel_size=3, padding=1)
        
        # Final upsampling layer to adjust the output size to (84x84)
        self.upsample_final = nn.Upsample(size=(84, 84), mode='bilinear', align_corners=True)
        
    def forward(self, x, encoder_features=None):
        # Incorporate skip connections if encoder_features are provided
        if encoder_features:
            x = self.up1(x)
            x = torch.cat([x, encoder_features[3]], dim=1)
            x = self.up2(x)
            x = torch.cat([x, encoder_features[2]], dim=1)
            x = self.up3(x)
            x = torch.cat([x, encoder_features[1]], dim=1)
            x = self.up4(x)
            x = torch.cat([x, encoder_features[0]], dim=1)
        else:
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
        
        x = self.conv_final(x)
        # Upsample to (84, 84)
        x = self.upsample_final(x)
        # Apply activation function
        x = torch.sigmoid(x)  # or torch.tanh(x) based on your input normalization
        return x

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)


def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True, method=None):
   if method == 'imaml_idcg':
      return ResNet_ImageNet(flatten = True, model = 'resnet18', frozen=True, layers=6)
   else:
      return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True, method=None):
    if method == 'imaml_idcg':
      return ResNet_ImageNet(flatten = True, model = 'resnet34', frozen=True, layers=6)
    else:
      return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True, method=None):
    if method == 'imaml_idcg':
        return ResNet_ImageNet(flatten = True, model = 'resnet50', frozen=True, layers=6)
    else:
        return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True, method=None):
    if method == 'imaml_idcg':
          return ResNet_ImageNet(flatten = True, model = 'resnet101', frozen=True, layers=6)
    else:
          return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)


