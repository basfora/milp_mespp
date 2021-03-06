import pytest

from core import plan_fun as pln
from core import construct_model as cm
from core import create_parameters as cp
from core import extract_info as ext
import numpy as np


def test_get_vertices_and_steps_start():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0_searchers = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0_searchers)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)
    assert start[0] == v0_searchers[0]
    assert start[1] == v0_searchers[1]


def test_get_vertices_and_steps_vertices():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0 = [1, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)
    assert vertices_t.get((1, 0)) == [1]
    assert vertices_t.get((1, 1)) == [1, 2, 3]
    assert vertices_t.get((1, 2)) == [1, 2, 3, 4, 5]
    assert vertices_t.get((1, 3)) == [1, 2, 3, 4, 5, 6]


def test_get_vertices_and_steps_vertices2():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)
    assert vertices_t.get((1, 0)) == [3]
    assert vertices_t.get((1, 1)) == [1, 3,  5]
    assert vertices_t.get((1, 2)) == [1, 2, 3, 5, 6]
    assert vertices_t.get((1, 3)) == [1, 2, 3, 4, 5, 6, 7]


def test_get_vertices_and_steps_times():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0 = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)
    assert times_v.get((1, 1)) == [1, 2, 3]
    assert times_v.get((1, 2)) == [2, 3]
    assert times_v.get((1, 3)) == [0, 1, 2, 3]
    assert times_v.get((1, 4)) == [3]
    assert times_v.get((1, 5)) == [1, 2, 3]
    assert times_v.get((1, 6)) == [2, 3]
    assert times_v.get((1, 7)) == [3]


def test_get_vertices_and_steps_distributed():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    v0 = [1, 2]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0)

    temp_s_path = pln.init_temp_path(searchers, deadline)
    temp_s_path['current_searcher'] = 1

    start, vertices_t, times_v = cm.get_vertices_and_steps_distributed(g, deadline, searchers, temp_s_path)

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


def test_neighbors():
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)

    v0 = [3, 1]
    deadline = 3
    # searchers
    searchers = cp.create_dict_searchers(g, v0)

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, deadline, searchers)

    s = 1
    v = 3
    t = 2
    tau_ext = ext.get_set_time_u_0(deadline)
    v_possible = cm.get_next_vertices(g, s, v, t, vertices_t, tau_ext)

    assert v_possible == [1, 5, 3]


# test previous aux_classes
def test_assemble_big_matrix():
    """test assemble of [1, 0; 0, M]"""

    n, b_0, M, searchers = parameters_7v_random_motion()

    big_M = cm.assemble_big_matrix(n, M)

    bigM = ext.convert_list_array(big_M, 'list')

    assert isinstance(bigM, list)
    assert bigM[0][0] == 1
    assert bigM[0][1] == 0
    assert bigM[0][2] == 0
    assert bigM[0][3] == 0
    assert bigM[0][4] == 0
    assert bigM[0][5] == 0
    assert bigM[0][6] == 0
    assert bigM[0][7] == 0

    assert bigM[1][0] == 0
    assert round(bigM[1][1], 2) == 0.33
    assert round(bigM[1][2], 2) == 0.33
    assert round(bigM[1][4], 2) == 0

    assert bigM[2][0] == 0
    assert bigM[3][0] == 0
    assert bigM[4][0] == 0
    assert bigM[5][0] == 0
    assert bigM[6][0] == 0
    assert bigM[7][0] == 0


def test_assemble_big_matrix2():
    """test assemble of [1, 0; 0, M]"""
    # input parameters for graph
    n_vertex = 7

    graph_name = 'G7V_test.p'
    g = ext.get_graph(graph_name)

    # input for target initial vertices (belief)
    v_target = [7]

    # type of motion
    target_motion = 'static'
    belief_distribution = 'uniform'
    b_0, M = cp.my_target_motion(g, v_target, belief_distribution, target_motion)

    big_M = cm.assemble_big_matrix(n_vertex, M)

    bigM = ext.convert_list_array(big_M, 'list')

    assert isinstance(bigM, list)
    assert bigM[0][0] == 1
    assert bigM[0][1] == 0
    assert bigM[0][2] == 0
    assert bigM[0][3] == 0
    assert bigM[0][4] == 0
    assert bigM[0][5] == 0
    assert bigM[0][6] == 0
    assert bigM[0][7] == 0

    assert bigM[1][0] == 0
    assert bigM[1][1] == 1
    assert bigM[2][2] == 1
    assert bigM[3][3] == 1
    assert bigM[4][4] == 1
    assert bigM[5][5] == 1
    assert bigM[6][6] == 1
    assert bigM[7][7] == 1

    assert bigM[2][0] == 0
    assert bigM[3][0] == 0
    assert bigM[4][0] == 0
    assert bigM[5][0] == 0
    assert bigM[6][0] == 0
    assert bigM[7][0] == 0


def test_change_type():
    A1 = np.array([[1, 2], [3, 4]])
    A2 = [1, 2, 3, 4, 5]

    B1 = ext.convert_list_array(A1, 'list')
    B2 = ext.convert_list_array(A2, 'array')

    B3 = ext.convert_list_array(A1, 'array')
    B4 = ext.convert_list_array(A2, 'list')

    assert isinstance(B1, list)
    assert isinstance(B2, np.ndarray)
    assert B3 is False
    assert B4 is False


def test_sample_vertex():

    my_vertices = [1, 2, 3, 4, 5, 6, 7]
    prob_move = [0.5, 0.5, 0, 0, 0, 0, 0]
    count1 = 0
    count2 = 0
    count3 = 0
    N = 10000
    i = 0
    while i < N:
        my_vertex = ext.sample_vertex(my_vertices, prob_move)
        if my_vertex == 1:
            count1 = count1 + 1
        elif my_vertex == 2:
            count2 = count2 + 1
        else:
            count3 += count3
        i = i + 1

    assert count3 == 0
    assert (count1 >= N/2 - 0.1 * N) and (count1 <= N/2 + 0.1 * N)
    assert (count2 >= N / 2 - 0.1 * N) and (count2 <= N / 2 + 0.1 * N)


def test_probability_move():

    n, b_0, M, searchers = parameters_7v_random_motion()

    prob_v = {}
    v = {}
    for i in range(1, 8):
        v[i], prob_v[i] = cm.probability_move(M, i)

    assert v[1] == [1, 2, 3]
    assert [round(i, 2) for i in prob_v[1]] == [0.33, 0.33, 0.33]
    assert v[2] == [1, 2, 4, 5]
    assert prob_v[2] == [0.25, 0.25, 0.25, 0.25]
    assert v[3] == [1, 3, 5]
    assert [round(i, 2) for i in prob_v[3]] == [0.33, 0.33, 0.33]
    assert v[4] == [2, 4]
    assert prob_v[4] == [0.5, 0.5]
    assert v[5] == [2, 3, 5, 6]
    assert prob_v[5] == [0.25, 0.25, 0.25, 0.25]
    assert v[6] == [5, 6, 7]
    assert [round(i, 2) for i in prob_v[6]] == [0.33, 0.33, 0.33]
    assert v[7] == [6, 7]
    assert prob_v[7] == [0.5, 0.5]


def test_product_capture_matrix():

    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    # initial searcher vertices
    v_searchers = [1]
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    # searchers
    searchers = cp.create_dict_searchers(g, v_searchers)

    s = 1
    v = 1
    t = 0

    C1 = searchers[s].get_capture_matrix(v)

    new_pos = dict()
    new_pos[s] = v

    prod_C = cm.product_capture_matrix(searchers, new_pos, 7)

    assert prod_C.all() == C1.all()


def test_product_capture_matrix2():

    n, b_0, M, searchers = parameters_7v_random_motion()

    C1 = cm.get_capture_s(searchers, 1, 1)
    C2 = cm.get_capture_s(searchers, 2, 2)
    correct_prod = C1 * C2

    new_pos = dict()
    new_pos[1] = 1
    new_pos[2] = 2
    t = 0

    prod_C = cm.product_capture_matrix(searchers, new_pos, n)

    assert prod_C.all() == correct_prod.all()


def test_belief_update_equation():

    n, b_0, M, searchers = parameters_7v_random_motion()

    new_pos = dict()
    new_pos[1] = 1
    new_pos[2] = 2

    # find the product of the capture matrices, s = 1...m
    prod_C = cm.product_capture_matrix(searchers, new_pos, n)

    # assemble the matrix for multiplication
    big_M = cm.assemble_big_matrix(n, M)

    new_belief = cm.belief_update_equation(b_0, big_M, prod_C)

    init_belief = ext.convert_list_array(b_0, 'array')
    my_belief = init_belief.dot(big_M).dot(prod_C)

    my_belief = my_belief.tolist()

    assert len(new_belief) == len(b_0)
    assert new_belief == my_belief


def test_get_true_position():

    # one possible vertex
    v1 = [1]
    pos1 = cm.get_target_true_position(v1)

    # more then one vertex, no idx provided
    v2 = [1, 2, 3]
    pos2 = cm.get_target_true_position(v2)
    pos3 = cm.get_target_true_position(v2)

    # more then one vertex, idx provided
    pos4 = cm.get_target_true_position(v2, 0)
    pos5 = cm.get_target_true_position(v2, 1)
    pos6 = cm.get_target_true_position(v2, 3)

    assert pos1 == v1[0]
    assert pos2 in set(v2)
    assert pos3 in set(v2)
    assert pos4 == v2[0]
    assert pos5 == v2[1]
    assert pos6 == v2[2]


def parameters_7v_random_motion():
    """Parameters pre-defined for unit tests"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    # input for target initial vertices (belief)
    v_target = [7]
    # initial searcher vertices
    v_searchers = [1, 2]
    deadline = 3
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    # initialize parameters
    b_0 = cp.set_initial_belief(g, v_target, belief_distribution)
    M = cp.set_motion_matrix(g, target_motion)
    searchers = cp.create_dict_searchers(g, v_searchers)

    n = 7
    return n, b_0, M, searchers

