#!/usr/bin/env bash
virtualenv --system-site-packages model_env
source model_env/bin/activate
pip install ipython
deactivate
