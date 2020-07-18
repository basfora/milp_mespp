# milp_mespp
## Overview
MILP models for the Multi-robot Efficient Search Path Planning (MESPP) problem: team of robots is deployed in a graph-represented environment to capture a moving target within a given deadline. 
Supports multiple searchers, arbitrary capture ranges, and false negatives simultaneously.

## Code Structure

   milp_mespp			<br />
   ├── classes			<br />
   │   ├── belief		<br />
   │   ├── inputs		<br />
   │   ├── searcher		<br />
   │   ├── solver_data		<br />
   │   └── target		<br />
   ├── core			<br />
   │   ├── extract_info		<br />
   │   ├── create_parameters	<br />
   │   ├── construct_model	<br />
   │   ├── milp_fun		<br />
   │   ├── plan_fun		<br />
   │   ├── sim_fun		<br />
   │   ├── retrieve_data	<br />
   │   └── plot_fun		<br />
   ├── data			<br />
   ├── examples			<br />
   │   ├── plan_only		<br />
   │   └── numerical_sim	<br />
   ├── graphs			<br />
   └── tests			<br />


## Installation Guide

This project was developed in Python 3.6. It uses the following Python libraries: datetime, sys, os, pickle, numpy, matplotlib, igraph, gurobipy.

Start by cloning this repository, 
```
git clone https://github.com/basfora/milp_mespp.git
```

> This project may be offered as a Python package in the future, but at the moment the code is still under construction. Please pull for updates every now and then, and report any bugs - I will do my best to fix them. 


### Installing gurobipy
Gurobi License and installation instructions [here](https://www.gurobi.com/documentation/9.0/quickstart_linux/index.html) <br />
> Important: changing default saving location of license file will cause errors! Don't do that. 

To install `gurobipy`, run (change path, OS and version accordingly),
```
cd path-to-folder/gurobi902/linux64/
sudo python3 setup.py install
```
> If you are using PyCharm, you might need to also run these commands on PyCharm's terminal.

### Run install script

This will install all the other necessary Python libraries and add the folder to your Python path system variable. From inside the `milp_mespp` folder, run on terminal:
```
./install_script.sh
``` 

Press ENTER and enter your user password when prompted.
> This script assumes Ubunty 18.04. For other OS the project code should work, but you will need to install the libraries/set path manually. 

When the installation is done, you should see this on your terminal (actual vertex numbers may vary):

```
--
Planned path: 
Searcher 1: [27, 54, 53, 58, 44, 45, 46, 47, 47, 47, 48]
t = 0
Target vertex: 13
Searcher 1: vertex 54 
```
(...)
```
--
t = 9
Target vertex: 49
Searcher 1: vertex 48 

```

This means both the planner and simulator are working and your installation is complete.

## Examples

To learn how to change specs and run multiple instances, check `examples/numerical_sim`.

> Data and plots will be saved in a milp_mespp/data folder (created the first time you run numerical_sim.py).


## Troubleshooting

If you try to run the `install_script.sh` and get the error `bash: ./install_script.sh: Permission denied`, make sure file *Properties > Permissions > Execute: Allow executing file as program* is checked

### Manual install
If you don't want to use the `install_script` or run into errors, you can install things manually.

#### Installing commonly used libraries
On terminal:
```
python3 -m pip install -U matplotlib
python3 -m pip install -U numpy
python3 -m pip install -U pytest
sudo apt-get install build-essential
sudo apt-get install python3.6-dev
```

#### Installing igraph
```
sudo add-apt-repository ppa:igraph/ppa
sudo apt-get update
sudo apt-get install python-igraph
```

If it throws errors, run: 
```
sudo apt-get install bison flex
python3 -m pip install python-igraph
```

#### Setting path

Add folder absolute path to your $PYTHONPATH system variable. On Linux OS, paste this on your `.bashrc` file (change path accordingly):

> export PYTHONPATH="${PYTHONPATH}:path-to-folder/milp_mespp"

Don't forget to source it (or restart your computer):
```
source ~/.bashrc
```

#### Run default simulator

Make sure things are working by running the simulator with default values.
```
cd milp_mespp/core
python3 sim_fun.py
```

You should see this in the terminal (actual vertex numbers may vary):

```
Planned path: 
Searcher 1: [27, 54, 53, 58, 44, 45, 46, 47, 47, 47, 48]
t = 0
Target vertex: 13
Searcher 1: vertex 54 
```

## Author
Beatriz Asfora

## Acknowledgements
Dr. Jacobo Banfi <br />
Prof. Mark Campbell


