import add_to_path
from core import construct_model as cm
from core import create_parameters as cp
from core import extract_info as ext
from core import milp_fun as mf
from gurobipy import *


def test_add_searcher_variables_x():
    """Test for expected X in simple graph"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0_searchers = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0_searchers)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)

    md = Model("my_model")

    var_for_test = mf.add_searcher_variables(md, g, start, vertices_t, deadline)[1]

    assert var_for_test.get('x')[0] == 'x[1,3,0]'

    assert var_for_test.get('x')[1] == 'x[1,1,1]'
    assert var_for_test.get('x')[2] == 'x[1,3,1]'
    assert var_for_test.get('x')[3] == 'x[1,5,1]'

    assert var_for_test.get('x')[4] == 'x[1,1,2]'
    assert var_for_test.get('x')[5] == 'x[1,2,2]'
    assert var_for_test.get('x')[6] == 'x[1,3,2]'
    assert var_for_test.get('x')[7] == 'x[1,5,2]'
    assert var_for_test.get('x')[8] == 'x[1,6,2]'

    assert var_for_test.get('x')[9] == 'x[1,1,3]'
    assert var_for_test.get('x')[10] == 'x[1,2,3]'
    assert var_for_test.get('x')[11] == 'x[1,3,3]'
    assert var_for_test.get('x')[12] == 'x[1,4,3]'
    assert var_for_test.get('x')[13] == 'x[1,5,3]'
    assert var_for_test.get('x')[14] == 'x[1,6,3]'
    assert var_for_test.get('x')[15] == 'x[1,7,3]'


def test_add_searcher_variables_y():
    """Test for expected Y in simple graph"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0_searchers = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0_searchers)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)

    md = Model("my_model")

    var_for_test = mf.add_searcher_variables(md, g, start, vertices_t, deadline)[1]

    assert var_for_test.get('y')[0] == 'y[1,3,1,0]'
    assert var_for_test.get('y')[1] == 'y[1,3,5,0]'
    assert var_for_test.get('y')[2] == 'y[1,3,3,0]'

    assert var_for_test.get('y')[3] == 'y[1,1,2,1]'
    assert var_for_test.get('y')[4] == 'y[1,1,3,1]'
    assert var_for_test.get('y')[5] == 'y[1,1,1,1]'

    assert var_for_test.get('y')[6] == 'y[1,3,1,1]'
    assert var_for_test.get('y')[7] == 'y[1,3,5,1]'
    assert var_for_test.get('y')[8] == 'y[1,3,3,1]'


def test_add_target_variables_b():
    """Test for expected B in simple graph"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    deadline = 3

    md = Model("my_model")

    var_for_test = mf.add_target_variables(md, g, deadline)[1]

    assert var_for_test.get('beta')[0] == '[0,0]'
    assert var_for_test.get('beta')[1] == '[1,0]'
    assert var_for_test.get('beta')[2] == '[2,0]'
    assert var_for_test.get('beta')[3] == '[3,0]'
    assert var_for_test.get('beta')[4] == '[4,0]'
    assert var_for_test.get('beta')[5] == '[5,0]'
    assert var_for_test.get('beta')[6] == '[6,0]'
    assert var_for_test.get('beta')[7] == '[7,0]'


def test_add_target_variables_alpha():
    """Test for expected B in simple graph"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    deadline = 3

    md = Model("my_model")

    var_for_test = mf.add_target_variables(md, g, deadline)[1]

    assert var_for_test.get('alpha')[0] == '[1,1]'
    assert var_for_test.get('alpha')[1] == '[2,1]'
    assert var_for_test.get('alpha')[2] == '[3,1]'

    assert var_for_test.get('alpha')[-3] == '[5,3]'
    assert var_for_test.get('alpha')[-2] == '[6,3]'
    assert var_for_test.get('alpha')[-1] == '[7,3]'


def test_get_var():
    """Test for expected B in simple graph"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0_searchers = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0_searchers)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)

    md = Model("my_model")
    # time indexes
    Tau_ = ext.get_idx_time(deadline)

    searchers_vars = mf.add_searcher_variables(md, g, start, vertices_t, deadline)[0]
    # variables related to target position belief and capture
    target_vars = mf.add_target_variables(md, g, deadline)[0]

    # get my variables together in one dictionary
    my_vars = {}
    my_vars.update(searchers_vars)
    my_vars.update(target_vars)

    my_chosen_var = mf.get_var(my_vars, 'x')
    my_empty_var = mf.get_var(my_vars, 'f')

    assert my_chosen_var == searchers_vars.get('x')
    assert my_empty_var is None


def test_position_searchers():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    # input for target initial vertices (belief)
    v_target = [7]
    # initial searcher vertices
    v0_searchers = [1, 2]
    horizon = 3
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    b0, M = cp.my_target_motion(g, v_target, belief_distribution, target_motion)
    # searchers
    searchers = cp.create_dict_searchers(g, v0_searchers)

    # solve
    # create model
    md = Model("my_model")

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers)

    # add variables
    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, )

    # add constraints (central algorithm)
    mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b0, M)

    mf.set_solver_parameters(md, 0.99, horizon, my_vars)

    md.update()
    # Optimize model
    md.optimize()

    x_s, b_target = mf.query_variables(md)

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
