# milp_mespp
## Overview
MILP models for the Multi-robot Efficient Search Path Planning (MESPP) problem: team of robots is deployed in a graph-represented environment to capture a moving target within a given deadline. 
Supports multiple searchers, arbitrary capture ranges, and false negatives simultaneously.


## Project Structure

 milp_mespp   <br />
   ├── core   <br />
   │   ├── extract_info  <br />
   │   ├── create_parameters  <br />
   │   ├── construct_model  <br />
   │   ├── milp_fun
   │   ├── plan_fun
   │   ├── sim_fun
   │   ├── retrieve_data
   │   └── plot_fun
   │
   ├── graphs
   ├── data
   └── examples
       ├── plan_only
       └── numerical_sim
   

## Installation Guide


### External Python Libraries
Project uses: datetime, os, pickle, random, numpy, matplotlib, igraph, gurobipy

### Installing commonly used libraries
Run on terminal:
```
python3 -m pip install -U matplotlib
python3 -m pip install -U numpy
python3 -m pip install -U random
sudo apt-get install build-essential
sudo apt-get install python3.6-dev
```

### Installing igraph
Run on terminal:
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
### Installing gurobipy
License and installation instructions [here].(https://www.gurobi.com/documentation/9.0/quickstart_linux/index.html) <br />
> Important: changing default saving location of license file will cause errors. 

If you are using PyCharm, you might need to also run on PyCharm's terminal:
```
cd path-to-folder/gurobi902/linux64/
sudo python3 setup.py install
```

## Troubleshooting




## Author
Beatriz Asfora

## Acknowledgements
Dr. Jacobo Banfi @github/jacoban <br />
Prof. Mark Campbell


