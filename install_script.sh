#!/bin/bash -i
# install python libraries
sudo apt-get install python3-distutils
sudo apt-get install python3-apt
sudo apt install python3-pip
python3 -m pip install -U matplotlib
python3 -m pip install -U numpy
python3 -m pip install -U pytest
sudo apt-get install build-essential
sudo apt-get install python3.8-dev
sudo apt-get install ffmpeg

# install igraph
sudo apt-get install bison flex
sudo add-apt-repository ppa:igraph/ppa
sudo apt-get update
sudo apt-get install python-igraph

# get milp_mespp path
MYSCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$MYSCRIPT")
# add to python path
echo '# milp_mespp path' >> ~/.bashrc
echo export PYTHONPATH=\"\${PYTHONPATH}:$SCRIPTPATH\" >> ~/.bashrc
source ~/.bashrc

# test with default simulation
cd core/
python3 sim_fun.py
