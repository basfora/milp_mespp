from core import milp_model_functions as mf
from core import create_parameters as cp


# input parameters for graph
n_vertex = 3
opt_graph = 2
graph_name = "3V3E"
g, deadline = cp.my_graph(n_vertex, graph_name, opt_graph)

# input for target initial vertices (belief)
v_target = [3]
b_0, M = cp.my_target_motion(g, v_target, None, 'static')

# initial searcher vertices
v_searchers = [1]
searchers_info = cp.my_searchers_info(g, v_searchers)

# save and show me
cp.save_and_plot_graph(g, graph_name, v_searchers, v_target)

# model
mf.run_gurobi(g, deadline, searchers_info, b_0, M)


# OBS: s \in {1,...m}
# G(V, E), v \in {1,...n} but igraph treats as zero
# t \in tau = {1,...T}
# capture matrix: first index is zero [0][0], first vertex is [1][1]
# v0 can be {1,...n} --> I convert to python indexing later
# array order: [s, t, v, u]
