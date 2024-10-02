# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import configs
from typing import Callable, Optional, Tuple


def identity(x):
    return x


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
   

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')


        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])



class SetDataset:
    def __init__(self, data_file, batch_size, transform, label_folder=None):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform, label_folder=label_folder)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, label_folder=None):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.label_folder = label_folder  # Path to segmentation mask labels folder

   
    def __getitem__(self,i):
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')

        # Load segmentation mask only if label_folder is provided
        if self.label_folder is not None:
            mask_name = os.path.basename(image_path).replace('.jpg', '_labels.npy')  # Assuming .jpg format for images
            mask_path = os.path.join(self.label_folder, mask_name)
            mask = np.load(mask_path)
            mask = torch.from_numpy(mask).long()  # Convert to tensor and appropriate type for segmentation
        else:
            mask = None  # No mask, handle main classification task only

        img = self.transform(img)
        target = self.target_transform(self.cl)
        
         # If mask is available, return it; otherwise, return just img and target
        if mask is not None:
            return img, mask, target
        else:
            return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
