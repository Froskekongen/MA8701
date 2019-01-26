#!/usr/bin/env bash
virtualenv --system-site-packages keras_venv
source keras_venv/bin/activate
pip install keras, ipython
deactivate
