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
# The following line produces the trained BiLSTM model
# Warning - Can take a long time (20-30 hours)

julia --project=. src/run_blstm.jl

# The following line produces the "BiLSTM" portions of Tables 1 & 2, as well as
# Figures 9, 10 and 11

julia --project=. src/out_blstm.jl
