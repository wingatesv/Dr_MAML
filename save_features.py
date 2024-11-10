import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py

import configs
import backbone
from data.datamgr import SimpleDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet


from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
import torch.multiprocessing as mp

def save_features(model, data_loader, outfile ):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    all_feats=None
    count=0
    for i, (x,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    params = parse_args('save_features')
    
    print(f'Applying StainNet stain normalisation......') if params.sn else print()

    assert params.method not in ['maml', 'maml_approx', 'anil', 'annemaml', 'tra_anil'], 'maml variants do not support save_feature and run'

    if 'Conv' in params.model:
      image_size = 84 
    else:
      image_size = 224

    split = params.split
          
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

    # Different dataset split
    elif params.dataset == 'BreaKHis_4x_2':
      if split == 'base':
          loadfile = configs.data_dir['BreaKHis_4x'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['BreaKHis_4x'] + split + '_2.json'
    elif params.dataset == 'BreaKHis_10x_2':
      if split == 'base':
          loadfile = configs.data_dir['BreaKHis_10x'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['BreaKHis_10x'] + split + '_2.json'
    elif params.dataset == 'BreaKHis_20x_2':
      if split == 'base':
          loadfile = configs.data_dir['BreaKHis_20x'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['BreaKHis_20x'] + split + '_2.json'
    elif params.dataset == 'BreaKHis_40x_2':
      if split == 'base':
          loadfile = configs.data_dir['BreaKHis_40x'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['BreaKHis_40x'] + split + '_2.json'

    elif params.dataset == 'ISIC_2':
      if split == 'base':
          loadfile = configs.data_dir['ISIC'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['ISIC'] + split + '_2.json'

    elif params.dataset == 'Smear_2':
      if split == 'base':
          loadfile = configs.data_dir['Smear'] + 'base_2.json' 
      else:
          loadfile  = configs.data_dir['Smear'] + split + '_2.json'

    elif params.dataset == 'BreaKHis_cross_IDC':
      if split == 'base':
          loadfile = configs.data_dir['BreaKHis_40x'] + 'base_2.json' 
      elif split == 'val':
           loadfile  = configs.data_dir['BreaKHis_40x'] + 'val_2.json'
      else:
           loadfile  = configs.data_dir['BCHI'] + 'novel.json'

    elif params.dataset == 'ISIC_cross_IDC':
      if split == 'base':
          loadfile = configs.data_dir['ISIC'] + 'base_2.json' 
      elif split == 'val':
           loadfile  = configs.data_dir['ISIC'] + 'val_2.json'
      else:
           loadfile  = configs.data_dir['BCHI'] + 'novel.json'

    elif params.dataset == 'Smear_cross_IDC':
      if split == 'base':
          loadfile = configs.data_dir['Smear'] + 'base_2.json' 
      elif split == 'val':
           loadfile  = configs.data_dir['Smear'] + 'val_2.json'
      else:
           loadfile  = configs.data_dir['BCHI'] + 'novel.json'
         


    else:
        raise ValueError(f"Unsupported dataset: {params.dataset}")

    if params.dataset == 'BreaKHis_cross_IDC':
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "BreaKHis40X_2", params.model, params.method)
    elif params.dataset == 'ISIC_cross_IDC':
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "ISIC_2", params.model, params.method)
    elif params.dataset == 'Smear_cross_IDC':
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, "Smear_2", params.model, params.method)
    else:
        checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)

    if params.train_aug :
        checkpoint_dir += f'_{params.train_aug}'
    if params.sn:
        checkpoint_dir += '_stainnet'
        
    if not params.method in ['baseline', 'baseline++'] :
        checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)

    else:
        modelfile   = get_best_file(checkpoint_dir)

    if params.save_iter != -1:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 

    datamgr = SimpleDataManager(image_size, batch_size = 64)
    data_loader = datamgr.get_data_loader(loadfile, aug = 'none', sn = params.sn)

    if params.method in ['relationnet', 'relationnet_softmax']:
        if params.model == 'Conv4': 
            model = backbone.Conv4NP()
        elif params.model == 'Conv6': 
            model = backbone.Conv6NP()
        else:
            model = model_dict[params.model]( flatten = False )
    elif params.method in ['maml' , 'maml_approx', 'anil', 'imaml_idcg']: 
       raise ValueError('MAML variants do not support save feature')
    else:
        model = model_dict[params.model]()

    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    model.eval()

    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
