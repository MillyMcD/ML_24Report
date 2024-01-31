#!/bin/bash

#create a virtual environment
virtualenv spotify_env

#source the virtual environment so you can jump in to it
source spotify_env/bin/activate

#now we pip install all of our packages
#this is done by using a requirements.txt file which stores them all
pip3 install --upgrade -r requirements.txt

cd src/sent2vec/.

pip install .

#now deactivate, our environment is ready for use
deactivate