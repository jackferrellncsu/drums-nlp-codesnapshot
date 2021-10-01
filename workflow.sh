#!/bin/sh
SEED=0

#Activate python env
source .venv/Scripts/activate

#Generates data into Data folder
python src/run_mlm_datapr.py -s $SEED

#Runs MLM conf preds process:
#Warning- Takes a very long time
python src/run_mlm_bert.py -s $SEED

#Processes results and produces graphs and report:
python src/out_mlm.py

####################
# The following line produces the trained BiLSTM model along with the results
# from testing the model on 5 different train/test splits (6-8 hours runtime)

julia --project=. src/run_blstm.jl

# The following line produces the "BiLSTM" portions of Tables 1 & 2, as well as
# Figures 5, 6 and 7

julia --project=. src/out_blstm.jl
