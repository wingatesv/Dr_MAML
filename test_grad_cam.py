import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time
import torch.multiprocessing as mp
from tqdm import tqdm
import sys

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from methods.anil import ANIL
from methods.imaml_idcg import IMAML_IDCG

from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

# Grad Cam Dependencies
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms



if __name__ == '__main__':
    mp.set_start_method('spawn')
    params = parse_args('grad_cam')
    print(f'Applying {params.sn} stain normalisation......') if params.sn else print()


    iter_num = 1

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    # --------------------------- METHOD 1 ---------------------------
    if params.method_1 == 'baseline':
        model_1 = BaselineFinetune( model_dict[params.model_1], **few_shot_params )
    elif params.method_1 == 'baseline++':
        model_1 = BaselineFinetune( model_dict[params.model_1], loss_type = 'dist', **few_shot_params )
    elif params.method_1 == 'protonet':
        model_1 = ProtoNet( model_dict[params.model_1], **few_shot_params )
    elif params.method_1 == 'matchingnet':
        model_1 = MatchingNet( model_dict[params.model_1], **few_shot_params )
    elif params.method_1 in ['relationnet', 'relationnet_softmax']:
        if params.model_1 == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model_1 == 'Conv6': 
            feature_model = backbone.Conv6NP
        else:
            feature_model = lambda: model_dict[params.model_1]( flatten = False )
        loss_type = 'mse' if params.method_1 == 'relationnet' else 'softmax'
        model_1           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )

    elif params.method_1 in ['maml' , 'maml_approx', 'anil', 'imaml_idcg']:

      backbone.ConvBlock.maml = True
      backbone.SimpleBlock.maml = True
      backbone.BottleneckBlock.maml = True
      backbone.ResNet.maml = True

      if params.method_1 in ['maml', 'maml_approx']:
        assert params.model_1 not in ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F'], 'maml do not support frozen feature extractor'
        model_1 = MAML(  model_dict[params.model_1], approx = (params.method_1 == 'maml_approx') , **few_shot_params )
     
      elif params.method_1 == 'anil':
        model_1 = ANIL(  model_dict[params.model_1], approx = False , **few_shot_params )

      elif params.method_1 == 'imaml_idcg':
        assert params.model_1 in ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F', 'ResNet18_IM', 'ResNet34_IM', 'ResNet50_IM'], 'IMAML_IDCG only support ImageNet pre-trained feature extractor'
        model_1 = IMAML_IDCG(  model_dict[params.model_1], approx = False , **few_shot_params )


    else:
       raise ValueError('Unknown method')

    model_1 = model_1.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model_1, params.method_1)
    if params.train_aug:
        checkpoint_dir += f'_{params.train_aug}'
    if params.sn:
        checkpoint_dir += '_stainnet'

    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_3way_5shot'


    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model_1.load_state_dict(tmp['state'])


    # --------------------------- METHOD 2 ---------------------------
    if params.method_2 == 'baseline':
        model_2 = BaselineFinetune( model_dict[params.model_2], **few_shot_params )
    elif params.method_2 == 'baseline++':
        model_2 = BaselineFinetune( model_dict[params.model_2], loss_type = 'dist', **few_shot_params )
    elif params.method_2 == 'protonet':
        model_2 = ProtoNet( model_dict[params.model_2], **few_shot_params )
    elif params.method_2 == 'matchingnet':
        model_2 = MatchingNet( model_dict[params.model_2], **few_shot_params )
    elif params.method_2 in ['relationnet', 'relationnet_softmax']:
        if params.model_2 == 'Conv4': 
            feature_model = backbone.Conv4NP
        elif params.model_2 == 'Conv6': 
            feature_model = backbone.Conv6NP
        else:
            feature_model = lambda: model_dict[params.model_2]( flatten = False )
        loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
        model_2           = RelationNet( feature_model, loss_type = loss_type , **few_shot_params )

    elif params.method_2 in ['maml' , 'maml_approx', 'anil', 'imaml_idcg']:

      backbone.ConvBlock.maml = True
      backbone.SimpleBlock.maml = True
      backbone.BottleneckBlock.maml = True
      backbone.ResNet.maml = True

      if params.method_2 in ['maml', 'maml_approx']:
        assert params.model_2 not in ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F'], 'maml do not support frozen feature extractor'
        model_2 = MAML(  model_dict[params.model_2], approx = (params.method_2 == 'maml_approx') , **few_shot_params )
     
      elif params.method_2 == 'anil':
        model_2 = ANIL(  model_dict[params.model_2], approx = False , **few_shot_params )

      elif params.method_2 == 'imaml_idcg':
        assert params.model_2 in ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F', 'ResNet18_IM', 'ResNet34_IM', 'ResNet50_IM'], 'IMAML_IDCG only support ImageNet pre-trained feature extractor'
        model_2 = IMAML_IDCG(  model_dict[params.model_2], approx = False , **few_shot_params )


    else:
       raise ValueError('Unknown method')

    model_2 = model_2.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model_2, params.method_2)
    if params.train_aug:
        checkpoint_dir += f'_{params.train_aug}'
    if params.sn:
        checkpoint_dir += '_stainnet'

    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_3way_5shot' 

    if not params.method in ['baseline', 'baseline++'] : 
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        else:
            modelfile   = get_best_file(checkpoint_dir)
        if modelfile is not None:
            tmp = torch.load(modelfile)
            model_2.load_state_dict(tmp['state'])

    # ------------------------------------------------------

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split





    if params.method_1 in ['maml', 'maml_approx', 'anil', 'imaml_idcg'] and params.method_2 in ['maml', 'maml_approx', 'anil', 'imaml_idcg']: #maml do not support testing with feature
        if 'Conv' in params.model_1 or 'Conv' in params.model_2:
            image_size = 84 
        else:
            image_size = 224
     
        datamgr  = SetDataManager(image_size, n_eposide = iter_num, n_query = 0 , **few_shot_params)

        if params.dataset == 'BreaKHis_4x':
          if split == 'base':
              loadfile = configs.data_dir['BreaKHis_4x'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['BreaKHis_4x'] + split + '.json'
        elif params.dataset == 'BreaKHis_10x':
          if split == 'base':
              loadfile = configs.data_dir['BreaKHis_10x'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['BreaKHis_10x'] + split + '.json'
        elif params.dataset == 'BreaKHis_20x':
          if split == 'base':
              loadfile = configs.data_dir['BreaKHis_20x'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['BreaKHis_20x'] + split + '.json'
        elif params.dataset == 'BreaKHis_40x':
          if split == 'base':
              loadfile = configs.data_dir['BreaKHis_40x'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['BreaKHis_40x'] + split + '.json'

        elif params.dataset == 'ISIC':
          if split == 'base':
              loadfile = configs.data_dir['ISIC'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['ISIC'] + split + '.json'

        elif params.dataset == 'Smear':
          if split == 'base':
              loadfile = configs.data_dir['Smear'] + 'base.json' 
          else:
              loadfile  = configs.data_dir['Smear'] + split + '.json'


        else:
            raise ValueError(f"Unsupported dataset: {params.dataset}")

        novel_loader     = datamgr.get_data_loader( loadfile, aug = False, sn = params.sn)
        # get a batch of images
        images, _ = next(iter(novel_loader))
        images = images.view(1, 3, 224, 224)



        # create a list of image grids, one for each sample in the batch
        image_grids = []
        for i in range(images.shape[0]):
            grid = vutils.make_grid(images[i], normalize=True)
            image_grids.append(grid)

        # stack the image grids vertically to create a single image grid
        merged_grid = torch.cat(image_grids, dim=1)

        # display the merged grid of images
        plt.imshow(merged_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig('/content/original_image.png', bbox_inches='tight', pad_inches=0)

        merged_grid = merged_grid.permute(1, 2, 0)
        np_images = merged_grid.numpy()

        
        method_names =  [params.method_1, params.method_2]
        model_names = [params.model_1, params.model_2]

        for i, model in enumerate([model_1, model_2]):
            model.eval()
            #get target layer
            last_conv_layer = None

            if model_names[i] not in  ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F', 'ResNet18_IM', 'ResNet34_IM', 'ResNet50_IM']:
                for m in model.feature.trunk.modules():
                    if isinstance(m, nn.Conv2d):
                        last_conv_layer = m
                target_layer = [last_conv_layer]

            elif model_names[i] in  ['ResNet18_IM', 'ResNet34_IM', 'ResNet50_IM']:
                for m in model.feature.feature_extractor.modules():
                    if isinstance(m, nn.Conv2d):
                        last_conv_layer = m
                target_layer = [last_conv_layer]

            elif model_names[i]  in  ['ResNet18_IM_F', 'ResNet34_IM_F', 'ResNet50_IM_F']:
                # unfreeze the parameters for gradcam
                for param in model.feature.feature_extractor.parameters():
                     param.requires_grad = True

                for m in model.feature.feature_extractor.modules():
                    if isinstance(m, nn.Conv2d):
                        last_conv_layer = m
                target_layer = [last_conv_layer]

            # Construct the CAM object once, and then re-use it on many images:
            cam = HiResCAM(model=model, target_layers=target_layer, use_cuda=True)
            grayscale_cam = cam(input_tensor=images, targets=None)


            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(np_images, grayscale_cam, use_rgb=True)
            plt.imshow(np.transpose(visualization, (0, 1, 2)))

            vis_dir = f'{method_names[i]}_{model_names[i]}'
            if params.train_aug:
                vis_dir += f'_{params.train_aug}'
            if params.sn:
                vis_dir += '_stainnet'

            plt.savefig(f'vis_{vis_dir}.png', bbox_inches='tight', pad_inches=0)



        

