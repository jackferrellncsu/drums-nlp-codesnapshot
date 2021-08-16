#!/bin/sh
SEED=1

#Activate python env
source .venv/Scripts/activate

#Generates data into Data folder
python src/Code_Snapshot/RunFile_mlm_datapr.py -s $SEED

#Runs MLM conf preds process:
#Warning- Takes a very long time
python src/Code_Snapshot/RunFile_mlm_bert.py -s $SEED

