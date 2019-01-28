#!/usr/bin/env bash
virtualenv --system-site-packages model_env
source model_env/bin/activate
pip install -I ipython
pip install -I keras
pip install kaggle
deactivate
