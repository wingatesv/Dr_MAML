import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import math
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import configs

import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.anil import ANIL
from methods.imaml_idcg import IMAML_IDCG
from methods.sharpmaml import SharpMAML
from sam import SAM


import torch.multiprocessing as mp
from io_utils import model_dict, parse_args, get_resume_file  



def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    

   

    if optimization == 'Adam':
        # sharpness aware minimisation
        if params.method  == 'sharpmaml':
          print('Using SAM Optimizer.....')
          base_optimizer = torch.optim.Adam 
          optimizer = SAM(model.parameters(), base_optimizer, lr=0.001)

        else:
          if hasattr(model, 'task_lr'):
              learning_rate = 0.00001
              print(f'With Adaptive Learnable Learning rate, Adam LR:{learning_rate}')
              model.define_task_lr_params()
              model_params = list(model.parameters()) + list(model.task_lr.values())
              optimizer = torch.optim.Adam(model_params, lr=learning_rate)
          else:
              learning_rate = 0.0001
              print(f'With scalar Learning rate, Adam LR:{learning_rate}')
              optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



    else:
       raise ValueError('Unknown optimization, please define by yourself')
  
    max_acc = 0   
    total_training_time = 0
    val_acc = []    
    scheduler = None
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(stop_epoch-start_epoch), eta_min=0.0001)


    for epoch in range(start_epoch,stop_epoch):
        start_time = time.time() # record start time
        model.train()
        model.train_loop(epoch, base_loader,  optimizer) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop(val_loader)
        val_acc.append(acc)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            if hasattr(model, 'task_lr'):
                torch.save({'epoch':epoch, 'state':model.state_dict(), 'task_lr': model.task_lr}, outfile)
            else:
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)


        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            if hasattr(model, 'task_lr'):
                torch.save({'epoch':epoch, 'state':model.state_dict(), 'task_lr': model.task_lr}, outfile)
            else:
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

            
        elapsed_time = time.time() - start_time # calculate elapsed time
        total_training_time += elapsed_time
      
        
    elapsed_hours = total_training_time / 3600.0 # convert to hours
    print(f"Total Training Time: {elapsed_hours:.2f} h") # print elapsed time for current epoch in hours


    plt.plot(range(start_epoch, stop_epoch), val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.savefig(f'{params.checkpoint_dir}_val_acc.png')
    return model

if __name__=='__main__':
    mp.set_start_method('spawn')
    np.random.seed(10)
    params = parse_args('train')

    #  Cross Domain from BreaKHis to BCHI 
    if params.dataset == 'cross_IDC_4x':
        base_file = configs.data_dir['BreaKHis_4x'] + 'base.json' 
        val_file   = configs.data_dir['BCHI'] + 'val.json' 
    elif params.dataset == 'cross_IDC_10x':
        base_file = configs.data_dir['BreaKHis_10x'] + 'base.json' 
        val_file   = configs.data_dir['BCHI'] + 'val.json' 
    elif params.dataset == 'cross_IDC_20x':
        base_file = configs.data_dir['BreaKHis_20x'] + 'base.json' 
        val_file   = configs.data_dir['BCHI'] + 'val.json' 
    elif params.dataset == 'cross_IDC_40x':
        base_file = configs.data_dir['BreaKHis_40x'] + 'base.json' 
        val_file   = configs.data_dir['BCHI'] + 'val.json' 

    #  Cross Domain from BreaKHis to PathoIDC 40x 
    elif params.dataset == 'cross_IDC_4x_2':
        base_file = configs.data_dir['BreaKHis_4x'] + 'base.json' 
        val_file   = configs.data_dir['PathoIDC_40x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_10x_2':
            base_file = configs.data_dir['BreaKHis_10x'] + 'base.json' 
            val_file   = configs.data_dir['PathoIDC_40x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_20x_2':
        base_file = configs.data_dir['BreaKHis_20x'] + 'base.json' 
        val_file   = configs.data_dir['PathoIDC_40x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_40x_2':
            base_file = configs.data_dir['BreaKHis_40x'] + 'base.json' 
            val_file   = configs.data_dir['PathoIDC_40x'] + 'val.json' 

    #  Cross Domain from BreaKHis to PathoIDC 20x 
    elif params.dataset == 'cross_IDC_4x_3':
        base_file = configs.data_dir['BreaKHis_4x'] + 'base.json' 
        val_file   = configs.data_dir['PathoIDC_20x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_10x_3':
            base_file = configs.data_dir['BreaKHis_10x'] + 'base.json' 
            val_file   = configs.data_dir['PathoIDC_20x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_20x_3':
        base_file = configs.data_dir['BreaKHis_20x'] + 'base.json' 
        val_file   = configs.data_dir['PathoIDC_20x'] + 'val.json' 
    elif params.dataset == 'cross_IDC_40x_3':
            base_file = configs.data_dir['BreaKHis_40x'] + 'base.json' 
            val_file   = configs.data_dir['PathoIDC_20x'] + 'val.json' 

    #  BreaKHis long tail distribution problem
    elif params.dataset == 'long_tail_4x':
        base_file = configs.data_dir['BreaKHis_4x'] + 'base_long.json' 
        val_file   = configs.data_dir['BreaKHis_4x'] + 'val_long.json' 
    elif params.dataset == 'long_tail_10x':
        base_file = configs.data_dir['BreaKHis_10x'] + 'base_long.json' 
        val_file   = configs.data_dir['BreaKHis_10x'] + 'val_long.json' 
    elif params.dataset == 'long_tail_20x':
        base_file = configs.data_dir['BreaKHis_20x'] + 'base_long.json' 
        val_file   = configs.data_dir['BreaKHis_20x'] + 'val_long.json' 
    elif params.dataset == 'long_tail_40x':
        base_file = configs.data_dir['BreaKHis_40x'] + 'base_long.json' 
        val_file   = configs.data_dir['BreaKHis_40x'] + 'val_long.json' 

    else:
        raise ValueError(f"Unsupported dataset: {params.dataset}")
         
    if 'Conv' in params.model:
      image_size = 84
    elif 'EffNet' in params.model:
      image_size = 480
    else:
      image_size = 224


    optimization = 'Adam'

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            params.stop_epoch = 50
     
        else: # other meta-learning methods
         params.stop_epoch = 100 

    print(f'Applying {params.train_aug} data augmentation ......')
    print(f'Applying StainNet stain normalisation......') if params.sn else print()
    
    if params.method in ['baseline', 'baseline++'] :
      base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
      base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug, sn = params.sn)
      val_datamgr     = SimpleDataManager(image_size, batch_size = 64)
      val_loader      = val_datamgr.get_data_loader( val_file, aug = 'none', sn = params.sn)
     
      if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
      elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx', 'anil', 'imaml_idcg', 'sharpmaml']:
       
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        base_datamgr = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader  = base_datamgr.get_data_loader( base_file , aug = params.train_aug,  sn = params.sn)
        
        val_datamgr = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader = val_datamgr.get_data_loader( val_file, aug = 'none', sn = params.sn) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor  

        if params.method == 'protonet':
            model = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )


        elif params.method in ['maml' , 'maml_approx', 'anil', 'imaml_idcg', 'sharpmaml']:
          backbone.ConvBlock.maml = True
          backbone.SimpleBlock.maml = True
          backbone.BottleneckBlock.maml = True
          backbone.ResNet.maml = True
          backbone.SqueezeNet.maml = True

          if params.method in ['maml', 'maml_approx']:
            model = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
       
          elif params.method == 'anil':
            model = ANIL(  model_dict[params.model], approx = False , **train_few_shot_params )

          elif params.method == 'imaml_idcg':
            assert params.model not in ['Conv4', 'Conv6','Conv4NP', 'Conv6NP', 'ResNet10'], 'imaml_idcg do not support non-ImageNet pretrained model'
            feature_backbone = lambda: model_dict[params.model]( flatten = True, method = params.method)
            model = IMAML_IDCG(  feature_backbone, approx = False , **train_few_shot_params )


          elif params.method == 'sharpmaml':
            model = SharpMAML(  model_dict[params.model], approx = False , **train_few_shot_params )
              
        else:
          raise ValueError('Unknown method')