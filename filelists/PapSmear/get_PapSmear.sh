#!/bin/bash
wget https://mde-lab.aegean.gr/images/stories/docs/smear2005.zip
unzip /content/smear2005.zip
python /content/Dr_MAML/filelists/PapSmear/prepare_PapSmear_data.py