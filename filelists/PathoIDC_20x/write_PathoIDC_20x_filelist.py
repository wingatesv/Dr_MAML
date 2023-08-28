# import configs
import random
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir

data_path = '/content/PathoIDC_20x/PathoIDC_20x' 
savedir = '/content/drive/MyDrive/PhD/IDC_Grading_FSL/filelists/PathoIDC_20x/'
dataset_list = ['val', 'novel']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append([join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


# split the folders to different splits accordingly
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
      
      split_point = len(classfile_list) // 2
      if 'val' in dataset:
        file_list = file_list + classfile_list[:split_point]
        label_list = label_list + np.repeat(i, split_point).tolist()

      if 'novel' in dataset:
        file_list = file_list + classfile_list[split_point:]
        label_list = label_list + np.repeat(i, len(classfile_list) - split_point).tolist()


    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
