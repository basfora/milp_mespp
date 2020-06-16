from core import extract_info as ext
from core import construct_model as cm
from core import create_parameters as cp
from core import milp_model_functions as mf
from core import analyze_results as ar
from gurobipy import *


def test_position_searchers():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    # input for target initial vertices (belief)
    v_target = [7]
    # initial searcher vertices
    v_searchers = [1, 2]
    horizon = 3
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    b0, M = cp.my_target_motion(g, v_target, belief_distribution, target_motion)
    searchers_info = cp.my_searchers_info(g, v_searchers)

    # solve
    # create model
    md = Model("my_model")

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers_info)

    # add variables
    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, )

    # add constraints (central algorithm)
    mf.add_constraints(md, g, my_vars, searchers_info, vertices_t, horizon, b0, M)

    mf.set_solver_parameters(md, 0.99, horizon, my_vars)

    md.update()
    # Optimize model
    md.optimize()

    x_s, b_target = ar.query_variables(md, searchers_info)

    # check searcher position (1)
    assert x_s.get((1, 1, 0)) == 1
    assert x_s.get((1, 3, 1)) == 1
    assert x_s.get((1, 5, 2)) == 1
    assert x_s.get((1, 6, 3)) == 1
    # check searcher position (2)
    assert x_s.get((2, 2, 0)) == 1
    assert x_s.get((2, 5, 1)) == 1
    assert x_s.get((2, 6, 2)) == 1
    assert x_s.get((2, 7, 3)) == 1

    # check target belief t = 0
    assert b_target.get((0, 0)) == 0
    assert b_target.get((1, 0)) == 0
    assert b_target.get((2, 0)) == 0
    assert b_target.get((3, 0)) == 0
    assert b_target.get((4, 0)) == 0
    assert b_target.get((5, 0)) == 0
    assert b_target.get((6, 0)) == 0
    assert b_target.get((7, 0)) == 1

    # check target belief t = 1
    assert b_target.get((0, 1)) == 0
    assert b_target.get((1, 1)) == 0
    assert b_target.get((2, 1)) == 0
    assert b_target.get((3, 1)) == 0
    assert b_target.get((4, 1)) == 0
    assert b_target.get((5, 1)) == 0
    assert b_target.get((6, 1)) == 0.5
    assert b_target.get((7, 1)) == 0.5

    # check target belief t = 2
    assert round(b_target.get((0, 2)), 3) == 0.583
    assert b_target.get((1, 2)) == 0
    assert b_target.get((2, 2)) == 0
    assert b_target.get((3, 2)) == 0
    assert b_target.get((4, 2)) == 0
    assert b_target.get((5, 2)) == 0
    assert b_target.get((6, 2)) == 0
    assert round(b_target.get((7, 2)), 3) == 0.417

    # check target belief t = 3
    assert b_target.get((0, 3)) == 1
    assert b_target.get((1, 3)) == 0
    assert b_target.get((2, 3)) == 0
    assert b_target.get((3, 3)) == 0
    assert b_target.get((4, 3)) == 0
    assert b_target.get((5, 3)) == 0
    assert b_target.get((6, 3)) == 0
    assert b_target.get((7, 3)) == 0

