#!/bin/sh
SEED=0

#All python dependencies may be found in requirement.txt
#All julia dependencies may be foind in Manifest.toml
#Creates new environment called ".venv"
python -m venv .venv

#Activates created environment
source .venv/Scripts/activate

#Installs required python packages to new environment
python -m pip install -r requirements.txt

#Creates data used in all methods
#Stores in Data folder
python src/run_mlm_datapr.py -s $SEED

#Runs MLM conformal prediction process 5 times as reported in paper.
#Takes several hours to complete.
python src/run_mlm_bert.py -s $SEED

#Processes results of MLM conformal prediction process.
#Creates the information presetned in tables 3 & 4, and figures 9, 10 , 11
python src/out_mlm.py

####################
# The following line produces the trained BiLSTM model along with the results
# from testing the model on 5 different train/test splits (6-8 hours runtime)

julia --project=. src/run_blstm.jl

# The following line produces the "BiLSTM" portions of Tables 1 & 2, as well as
# Figures 5, 6 and 7

julia --project=. src/out_blstm.jl
