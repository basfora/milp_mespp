import pytest

from core import construct_model as cm
from core import create_parameters as cp
from core import extract_info as ext
from core import sim_fun as sf
from gurobipy import *


def test_get_vertices_and_steps_start():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)
    assert start[0] == v0[0]
    assert start[1] == v0[1]


def test_get_vertices_and_steps_vertices():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [1, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)
    assert vertices_t.get((1, 0)) == [1]
    assert vertices_t.get((1, 1)) == [1, 2, 3]
    assert vertices_t.get((1, 2)) == [1, 2, 3, 4, 5]
    assert vertices_t.get((1, 3)) == [1, 2, 3, 4, 5, 6]


def test_get_vertices_and_steps_vertices2():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)
    assert vertices_t.get((1, 0)) == [3]
    assert vertices_t.get((1, 1)) == [1, 3,  5]
    assert vertices_t.get((1, 2)) == [1, 2, 3, 5, 6]
    assert vertices_t.get((1, 3)) == [1, 2, 3, 4, 5, 6, 7]


def test_get_vertices_and_steps_times():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)
    assert times_v.get((1, 1)) == [1, 2, 3]
    assert times_v.get((1, 2)) == [2, 3]
    assert times_v.get((1, 3)) == [0, 1, 2, 3]
    assert times_v.get((1, 4)) == [3]
    assert times_v.get((1, 5)) == [1, 2, 3]
    assert times_v.get((1, 6)) == [2, 3]
    assert times_v.get((1, 7)) == [3]


def test_get_vertices_and_steps_distributed():
    # load graph
    graph_file = 'G7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [1, 2]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    temp_s_path = sf.init_temporary_path(searchers_info, deadline)
    temp_s_path['current_searcher'] = 1

    start, vertices_t, times_v = cm.get_vertices_and_steps_distributed(g, deadline, searchers_info, temp_s_path)

    assert times_v.get((1, 1)) == [0, 1, 2, 3]
    assert times_v.get((1, 2)) == [1, 2, 3]
    assert times_v.get((1, 3)) == [1, 2, 3]
    assert times_v.get((1, 4)) == [2, 3]
    assert times_v.get((1, 5)) == [2, 3]
    assert times_v.get((1, 6)) == [3]
    assert times_v.get((1, 7)) == []
    assert times_v.get((1, 8)) == [4]

    assert times_v.get((2, 1)) == []
    assert times_v.get((2, 2)) == [0, 1, 2, 3]
    assert times_v.get((2, 3)) == []
    assert times_v.get((2, 4)) == []
    assert times_v.get((2, 5)) == []
    assert times_v.get((2, 6)) == []
    assert times_v.get((2, 7)) == []
    assert times_v.get((2, 8)) == [4]

    assert vertices_t.get((1, 0)) == [1]
    assert vertices_t.get((1, 1)) == [1, 2, 3]
    assert vertices_t.get((1, 2)) == [1, 2, 3, 4, 5]
    assert vertices_t.get((1, 3)) == [1, 2, 3, 4, 5, 6]
    assert vertices_t.get((1, 4)) == [8]

    assert vertices_t.get((2, 0)) == [2]
    assert vertices_t.get((2, 1)) == [2]
    assert vertices_t.get((2, 2)) == [2]
    assert vertices_t.get((2, 3)) == [2]
    assert vertices_t.get((2, 4)) == [8]


def test_add_searcher_variables_x():
    """Test for expected X in simple graph"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)

    md = Model("my_model")

    var_for_test = cm.add_searcher_variables(md, g, start, vertices_t, deadline)[1]

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
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v_searchers = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v_searchers)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)

    md = Model("my_model")

    var_for_test = cm.add_searcher_variables(md, g, start, vertices_t, deadline)[1]

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
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    deadline = 3

    md = Model("my_model")

    var_for_test = cm.add_target_variables(md, g, deadline)[1]

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
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    deadline = 3

    md = Model("my_model")

    var_for_test = cm.add_target_variables(md, g, deadline)[1]

    assert var_for_test.get('alpha')[0] == '[1,1]'
    assert var_for_test.get('alpha')[1] == '[2,1]'
    assert var_for_test.get('alpha')[2] == '[3,1]'

    assert var_for_test.get('alpha')[-3] == '[5,3]'
    assert var_for_test.get('alpha')[-2] == '[6,3]'
    assert var_for_test.get('alpha')[-1] == '[7,3]'


def test_get_var():
    """Test for expected B in simple graph"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)

    md = Model("my_model")
    # time indexes
    Tau_ = ext.get_idx_time(deadline)

    searchers_vars = cm.add_searcher_variables(md, g, start, vertices_t, deadline)[0]
    # variables related to target position belief and capture
    target_vars = cm.add_target_variables(md, g, deadline)[0]

    # get my variables together in one dictionary
    my_vars = {}
    my_vars.update(searchers_vars)
    my_vars.update(target_vars)

    my_chosen_var = cm.get_var(my_vars, 'x')
    my_empty_var = cm.get_var(my_vars, 'f')

    assert my_chosen_var == searchers_vars.get('x')
    assert my_empty_var is None


def test_neighbors():
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)

    v0 = [3, 1]
    deadline = 3
    # searchers info
    searchers_info = cp.my_searchers_info(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers_info)

    s = 1
    v = 3
    t = 2
    Tau_ext = ext.get_set_ext_time(deadline)
    v_possible = cm.get_next_vertices(g, s, v, t, vertices_t, Tau_ext)

    assert v_possible == [1, 5, 3]


# def test_3vertices_var_x():
#     # load graph
#     graph_file = 'G_3V3E.p'
#     g = cm.get_graph(graph_file)
#     # input for target initial vertices (belief)
#     v_target = [3]
#     deadline = 2
#     b_0, M = cp.my_target_motion(g, v_target, None, 'static')
#     # initial searcher vertices
#     v_searchers = [1]
#     searchers_info = cp.my_searchers_info(g, v_searchers)
#
#
#
#     # model
#     # mf.solve_model(g, deadline, searchers_info, b_0, M)