#!/bin/sh


ENV_FILE=.venv
if test -e ENV_FILE; then
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install -r requirements.txt

fi