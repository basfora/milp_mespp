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
It's advisable to create a virtual environment for this project:

```
cd milp_mespp
python3 -m venv vmilp
source vmilp/bin/activate
```

### Python Libraries
Project uses: datetime, sys, os, pickle, numpy, matplotlib, igraph, gurobipy

### Installing commonly used libraries
Run on terminal:
```
python3 -m pip install -U matplotlib
python3 -m pip install -U numpy
python3 -m pip install -U pytest
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
Gurobi License and installation instructions [here](https://www.gurobi.com/documentation/9.0/quickstart_linux/index.html) <br />
> Important: changing default saving location of license file will cause errors. 

To install gurobipy, run (change version and OS accordingly):
```
cd path-to-folder/gurobi902/linux64/
sudo python3 setup.py install
```
> If you are using PyCharm, you might need to also run on PyCharm's terminal

### Installing this package

Make sure you have the latest version of the libraries:
```
sudo apt-get update
```

From inside the `milp_mespp` folder:

```
sudo python3 setup.py install
```


## Troubleshooting

Make sure things are working by running the run the simulator with default values.
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

## Examples

To learn how to change specs and run multiple instances, run `examples/numerical_sim`.

> Data and plots will be saved in a milp_mespp/data folder (created the first time you run numerical_sim.py).

## Author
Beatriz Asfora

## Acknowledgements
Dr. Jacobo Banfi <br />
Prof. Mark Campbell


