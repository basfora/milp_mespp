import pytest
from igraph import *
import numpy as np

from core import extract_info as ext
from core import create_parameters as cp
from core import construct_model as cm


def test_belief_vector():
    """Test b_0 for vertices with equal probability"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)

    v_target_init = [5]
    b_0, M = cp.my_target_motion(g, v_target_init)
    assert b_0 == [0, 0, 0, 0, 0, 1, 0, 0]

    v_target_init = [1, 7]
    b_0, M = cp.my_target_motion(g, v_target_init)
    assert b_0 == [0, 1/2, 0, 0, 0, 0, 0, 1/2]


def test_belief_vector_prob():
    """Test b_0 for several vertices, user defined probability"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)

    v_target_init = [1, 5, 7]
    init_prob = [0.2, 0.5, 0.3]
    b_0, M = cp.my_target_motion(g, v_target_init, init_prob)
    assert b_0 == [0, 0.2, 0, 0, 0, 0.5, 0, 0.3]


def test_markovian_matrix():
    """Test Markovian matrix M for random motion"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = ext.get_graph(graph_file)

    v_target_init = [1, 5, 7]
    init_prob = [0.2, 0.5, 0.3]
    b_0, M = cp.my_target_motion(g, v_target_init, init_prob)

    a = 1 / 3
    b = 1 / 4
    c = 1 / 2
    M_exp = [[a, a, a, 0, 0, 0, 0],
             [b, b, 0, b, b, 0, 0],
             [a, 0, a, 0, a, 0, 0],
             [0, c, 0, c, 0, 0, 0],
             [0, b, b, 0, b, b, 0],
             [0, 0, 0, 0, a, a, a],
             [0, 0, 0, 0, 0, c, c]]

    assert M == M_exp


def test_capture_range():
    graph_file = 'G64V_grid.p'
    g = ext.get_graph(graph_file)

    v_target = [1, 2, 3]
    v_searchers = [5]
    target_motion = 'random'
    belief_distribution = 'uniform'
    capture_range = 1
    zeta = None

    b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution,
                                                capture_range, zeta)
    s = 1
    u = 1

    C = cm.get_capture_matrix(searchers_info, s, u)

    assert C[0][0] == 1
    assert C[1][0] == 1
    assert C[2][0] == 1
    assert C[9][0] == 1


def test_random_picking():

    my_seed_target = 2000

    my_list = list(range(1, 60))

    random_list_previous = []

    for m in range(0, 5):
        my_seed = my_seed_target + m
        for i in range(0, 20):
            random_list = cp.pick_pseudo_random(my_list, my_seed, 5)
            if m != 0 and i != 0:
                dif_el = (set(random_list) - set(random_list_previous))
                assert len(dif_el) == 0
            # iterate
            random_list_previous = random_list


def test_check_reachability():
    g = ext.get_graph_02()
    capture_range = 1
    v_target = [2]
    v_searchers = [3, 9]
    v_searchers1 = [4, 9]
    v_searchers2 = [4, 10]

    init_is_ok = cp.check_reachability(g, capture_range, v_target, v_searchers)
    init_is_ok1 = cp.check_reachability(g, capture_range, v_target, v_searchers1)
    init_is_ok2 = cp.check_reachability(g, capture_range, v_target, v_searchers2)

    assert init_is_ok is False
    assert init_is_ok1 is True
    assert init_is_ok2 is True
    assert init_is_ok2 is True



