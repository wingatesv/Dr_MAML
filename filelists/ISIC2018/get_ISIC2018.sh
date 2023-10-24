#!/bin/bash
wget https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
unzip /content/ISIC2018_Task3_Training_Input.zip 
python /content/Dr_MAML/filelists/ISIC2018/prepare_data.py