#!/bin/bash

#activate the environment
source spotify_env/bin/activate

#create notebook
jupyter notebook --ip 0.0.0.0 --port 9001 --no-browser --allow-root &