# IMAML-IDCG: Gradient-Based Meta-Learning with ImageNet Feature Reusing for Few-Shot Invasive Ductal Carcinoma Grading

This repo contains the reference source code for the paper [IMAML-IDCG: Gradient-Based Meta-Learning with ImageNet Feature Reusing for Few-Shot Invasive Ductal Carcinoma Grading]


## Citation
If you find our code useful, please consider citing our work using the bibtex:

## Enviroment
 - Google Colab
 - Python3
 - [Pytorch](http://pytorch.org/) 
 - json

## Getting started
### Clone the Repo
* Clone the repo into your Google Colab working directory
<pre>
!git clone https://github.com/wingatesv/IMAML-IDCG.git
</pre>

### Download Datasets
* Please contact the author for the dataset link at: wingatesvoon@1utar.my

| Split |     Dataset     | 
|-------|-----------------|
| Base  | BreaKHis_4x     | 
|       | BreaKHis_10x    | 
|       | BreaKHis_20x    | 
|       | BreaKHis_40x    | 
| Novel | BCHI            |
|       | PathoIDC_20x    | 
|       | PathoIDC_40x    | 

### BreaKHis_4x
* Change directory to `./filelists/BreaKHis_4x`
* run `source ./get_BreaKHis_4x.sh`

### BreaKHis_10x
* Change directory to `./filelists/BreaKHis_10x`
* run `source ./get_BreaKHis_10x.sh`

### BreaKHis_20x
* Change directory to `./filelists/BreaKHis_20x`
* run `source ./get_BreaKHis_20x.sh`

### BreaKHis_40x
* Change directory to `./filelists/BreaKHis_40x`
* run `source ./get_BreaKHis_40x.sh`

### BCHI
* Change directory to `./filelists/BCHI`
* run `source ./get_BCHI.sh`

### PathoIDC_20x
* Change directory to `./filelists/PathoIDC_20x`
* run `source ./get_PathoIDC_20x.sh`

### PathoIDC_40x
* Change directory to `./filelists/PathoIDC_40x`
* run `source ./get_PathoIDC_40x.sh`


### BreaKHis_{}X->BCHI
* Finish preparation for BreaKHis and BCHI and you are done!

### BreaKHis_{}x->PathoIDC{}x
* Finish preparation for BreaKHis and PathoIDC and you are done!


### Self-defined setting
* * Require one data split json file: 'base.json' for each BreaKHis dataset
* * Require two data split json file: 'val.json', 'novel.json' for BCHI and PathoIDC datasets  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  

* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  

## Cross-Domain Configurations
| Name | Description |
|------|-------------|
| cross_IDC_4x | BreaKHis_4x to BCHI |
| cross_IDC_10x | BreaKHis_10x to BCHI |
| cross_IDC_20x | BreaKHis_20x to BCHI |
| cross_IDC_40x | BreaKHis_40x to BCHI |
| cross_IDC_4x_2 | BreaKHis_4x to PathoIDC_40x |
| cross_IDC_10x_2 | BreaKHis_10x to PathoIDC_40x  |
| cross_IDC_20x_2 | BreaKHis_20x to PathoIDC_40x  |
| cross_IDC_40x_2 | BreaKHis_40x to PathoIDC_40x  |
| cross_IDC_4x_3 | BreaKHis_4x to PathoIDC_20x |
| cross_IDC_10x_3 | BreaKHis_10x to PathoIDC_20x  |
| cross_IDC_20x_3 | BreaKHis_20x to PathoIDC_20x  |
| cross_IDC_40x_3 | BreaKHis_40x to PathoIDC_20x  |


## Train
Run
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python ./train.py --dataset cross_IDC_40x --model ResNet34 --method maml --train_n_way 3 --test_n_way 3 --n_shot 1 --stop_epoch 100 --train_aug --sn stainnet`  
Commands below follow this example, and please refer to io_utils.py for additional options.

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
Run
```python ./save_features.py --dataset cross_IDC_40x --model ResNet34 --method relationnet  --train_n_way 3 --n_shot 5 --test_n_way 3 --train_aug --sn stainnet```

## Test
Run
```python ./test.py --dataset cross_IDC_40x --model ResNet34 --method maml --train_n_way 3 --test_n_way 3 --n_shot 1 --train_aug --sn stainnet```

## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

## References
* Main Framework
https://github.com/wyharveychen/CloserLookFewShot
* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate 
* Omniglot dataset, Method: Prototypical Network
https://github.com/jakesnell/prototypical-networks
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml  
https://github.com/dragen1860/MAML-Pytorch  
https://github.com/katerakelly/pytorch-maml

