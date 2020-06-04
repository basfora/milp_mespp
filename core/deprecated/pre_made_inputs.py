from core import extract_info as ext
from core import create_parameters as cp
from igraph import *
import pickle

import sys


def get_sim_param(sim_param=1):
    """Get parameters related to this simulation run"""

    theta = None
    deadline = None
    horizon = None
    solver_type = None

    if sim_param == 1:
        theta = 2
        deadline = 6
        horizon = 3
        solver_type = 'central'
    elif sim_param == 2:
        theta = 3
        deadline = 6
        horizon = 3
        solver_type = 'central'
    elif sim_param == 3:
        theta = 3
        deadline = 6
        horizon = 3
        solver_type = 'distributed'
    elif sim_param == 4:
        theta = 10
        deadline = 10
        horizon = 10
        solver_type = 'central'
    elif sim_param == 5:
        theta = 20
        deadline = 20
        horizon = 20
        solver_type = 'central'
    else:
        print("No other options available at this time")
        exit()

    return horizon, theta, deadline, solver_type


def get_inputs(opt=1, m=1):
    """Pre-determined inputs for planning"""

    g = None
    v_target = None
    v_searchers = None
    target_motion = None
    belief_distribution = None

    if opt == 1:
        # graph is the 7V, 7E of Jacopo's paper
        graph_name = 'G7V7E.p'

        g = ext.get_graph(graph_name)
        g["name"] = 'G_7V7E'

        # target: random motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        # input for target initial ve
        v_target = [7]
        # initial searcher vertices
        v_searchers = [1, 2]
    elif opt == 2:
        # graph is the 7V, 7E of Jacopo's paper
        graph_name = 'G7V7E.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # horizon is 3

        # target: static motion, uniform initial distribution
        target_motion = 'static'
        belief_distribution = 'uniform'

        # input for target initial ve
        v_target = [7]
        # initial searcher vertices
        v_searchers = [1, 2]
    elif opt == 4:
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        horizon = 10

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        # input for target initial vertex
        v_target = [7]
        # initial searcher vertices
        v_searchers = [1]
    elif opt == 5:
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        horizon = 10

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        # input for target initial vertex
        v_target = [7]
        # initial searcher vertices
        v_searchers = [1, 2]
    elif opt == 6:
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        # input for target initial vertex
        v_target = [7]
        # initial searcher vertices
        v_searchers = [1, 2, 3]
    elif opt == 7:
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        # input for target initial vertex
        v_target = [7]
        # initial searcher vertices
        v_searchers = [15, 11, 25, 34, 33]
    elif opt == 8:
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        v_searchers, v_possible = cp.searcher_random_pos(g, m)
        v_target = cp.target_random_pos(v_possible, 'sure')
    elif opt == 9:
        # two searchers
        m = 2
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        v_searchers, v_possible = cp.searcher_random_pos(g, m)
        v_target = cp.target_random_pos(v_possible, 'sure')
    elif opt == 10:
        # three searchers
        m = 3
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        v_searchers, v_possible = cp.searcher_random_pos(g, m)
        v_target = cp.target_random_pos(v_possible, 'sure')
    elif opt == 11:
        # four searchers
        m = 4
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        v_searchers, v_possible = cp.searcher_random_pos(g, m)
        v_target = cp.target_random_pos(v_possible, 'sure')
    elif opt == 12:
        # 5 searchers
        m = 5
        # graph from (Hollinger, 2009)
        graph_name = 'G60V.p'

        g = ext.get_graph(graph_name)
        g["name"] = graph_name

        # target: static motion, uniform initial distribution
        target_motion = 'random'
        belief_distribution = 'uniform'

        v_searchers, v_possible = cp.searcher_random_pos(g, m)
        v_target = cp.target_random_pos(v_possible, 'sure')
    else:
        print("No other options available at this time")
        exit()

    return g, v_target, v_searchers, target_motion, belief_distribution







