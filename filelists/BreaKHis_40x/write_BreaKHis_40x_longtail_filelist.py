import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random

cwd = os.getcwd() 
data_path = '/content/BreaKHis_40x/BreaKHis_400x'
savedir = '/content/'
dataset_list = ['base','val','novel']

folder_list = ['fibroadenoma', 'mucinous_carcinoma', 'ductal_carcinoma', 'tubular_adenoma', 'adenosis', 'phyllodes_tumor', 'papillary_carcinoma', 'lobular_carcinoma']
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])

base_classes = ['ductal_carcinoma', 'fibroadenoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'lobular_carcinoma']
val_novel_classes = ['phyllodes_tumor', 'tubular_adenoma', 'adenosis']

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if folder_list[i] in base_classes and dataset == 'base':
            file_list += classfile_list
            label_list += np.repeat(i, len(classfile_list)).tolist()
        elif folder_list[i] in val_novel_classes:
            half_index = len(classfile_list) // 2
            if dataset == 'val':
                file_list += classfile_list[:half_index]
                label_list += np.repeat(i, half_index).tolist()
            elif dataset == 'novel':
                file_list += classfile_list[half_index:]
                label_list += np.repeat(i, len(classfile_list) - half_index).tolist()

    with open(join(savedir, f"{dataset}.json"), "w") as fo:
        json.dump({"label_names": folder_list, "image_names": file_list, "image_labels": label_list}, fo)

    print(f"{dataset} - OK")
