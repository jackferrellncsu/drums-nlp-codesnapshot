#!/bin/sh

#This file initializes a python virtual environment containing the required packages
#Must be run before workflow.sh
#Requires file "requirements.txt"

python -m venv .venv

source .venv/Scripts/activate

python -m pip install -r requirements.txt

