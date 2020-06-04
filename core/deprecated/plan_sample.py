from core import milp_model_functions as mf
from core import create_parameters as cp
from core import analyze_results as ar


# input parameters for graph
n_vertex = 7
opt_graph = 1
graph_name = "V7E7"
horizon = 3

# create directory to save graphs
name_folder = "0705_04"
full_name_folder = "data/" + name_folder
g, deadline = cp.my_graph(n_vertex, graph_name, opt_graph, horizon)

# input for target initial vertices (belief)
v_target = [7]
# initial searcher vertices
v_searchers = [1, 2]

# type of motion
target_motion = 'static'
belief_distribution = 'uniform'

# initialize parameters
b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution)
# model
results, md = mf.run_gurobi(g, full_name_folder, deadline, searchers_info, b_0, M)
if results is True:
    ar.show_me_results(md, g, name_folder, searchers_info, deadline)

# OBS: s \in {1,...m}
# G(V, E), v \in {1,...n} but igraph treats as zero
# t \in tau = {1,...T}
# capture matrix: first index is zero [0][0], first vertex is [1][1]
# v0 can be {1,...n} --> I convert to python indexing later
# array order: [s, t, v, u]
