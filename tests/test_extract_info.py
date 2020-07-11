import pytest
from core import extract_info as ext
from core import create_parameters as cp


def test_get_set_and_idx_searchers():
    list_input = [5, 2]
    S, m = ext.get_set_searchers(list_input)
    S_, m_ = ext.get_idx_searchers(list_input)
    assert S == [1, 2]
    assert S_ == [0, 1]
    assert m == 2
    assert m_ == 2

    dict_input = {1: 10, 2: 9, 3: 8}
    S, m = ext.get_set_searchers(dict_input)
    S_, m_ = ext.get_idx_searchers(dict_input)
    assert S == [1, 2, 3]
    assert S_ == [0, 1, 2]
    assert m == 3
    assert m_ == 3

    dict_input2 = dict()
    # x_s(s, v, t)
    dict_input2[(1, 1, 0)] = 1
    dict_input2[(1, 2, 1)] = 1
    dict_input2[(1, 3, 2)] = 1
    dict_input2[(2, 4, 0)] = 1
    dict_input2[(2, 5, 1)] = 1
    dict_input2[(2, 6, 2)] = 1
    S, m = ext.get_set_searchers(dict_input2)
    S_, m_ = ext.get_idx_searchers(dict_input2)
    assert S == [1, 2]
    assert S_ == [0, 1]
    assert m == 2
    assert m_ == 2

    int_input = 5
    S, m = ext.get_set_searchers(int_input)
    S_, m_ = ext.get_idx_searchers(int_input)
    assert S == [1, 2, 3, 4, 5]
    assert S_ == [0, 1, 2, 3, 4]
    assert m == 5
    assert m_ == 5


def test_get_m_h_from_tuple():

    # searcher position
    x_s = dict()
    # (s, v, t)
    x_s[(1, 1, 0)] = 1
    x_s[(1, 3, 1)] = 1
    x_s[(1, 5, 2)] = 1
    x_s[(1, 6, 3)] = 1

    x_s[(2, 2, 0)] = 1
    x_s[(2, 5, 1)] = 1
    x_s[(2, 6, 2)] = 1
    x_s[(2, 7, 3)] = 1

    # searchers path
    path = dict()
    # (s, t)
    path[(1, 0)] = 1
    path[(1, 1)] = 3
    path[(1, 2)] = 5
    path[(1, 3)] = 6

    path[(1, 0)] = 2
    path[(2, 1)] = 5
    path[(2, 2)] = 6
    path[(2, 3)] = 7

    m = ext.get_m_from_tuple(x_s)
    m2 = ext.get_m_from_tuple(path)
    h = ext.get_h_from_tuple(x_s)
    h2 = ext.get_h_from_tuple(path)

    assert m == 2
    assert m2 == 2
    assert h == 3
    assert h2 == 3


def test_get_searchers_positions():

    g = ext.get_graph_00()
    v0_searchers = [1, 2]

    searchers = cp.create_dict_searchers(g, v0_searchers)

    s_pos = ext.get_searchers_positions(searchers)
    assert s_pos == [1, 2]


def test_get_time():

    deadline = 3
    T = ext.get_set_time(deadline)
    assert T == [1, 2, 3]

    T_idx = ext.get_idx_time(deadline)
    assert T_idx == [0, 1, 2]

    T_ext = ext.get_set_time_u_0(deadline)
    T_ext2 = ext.get_set_time_u_0(T)
    assert T_ext == [0, 1, 2, 3]
    assert T_ext2 == [0, 1, 2, 3]


def test_get_vertices():

    g = ext.get_graph_00()

    V, n = ext.get_set_vertices(g)
    assert V == [1, 2, 3, 4, 5, 6, 7]
    assert n == 7

    V, n = ext.get_set_vertices(n)
    assert V == [1, 2, 3, 4, 5, 6, 7]
    assert n == 7

    V_idx, n = ext.get_idx_vertices(g)
    assert V_idx == [0, 1, 2, 3, 4, 5, 6]
    assert n == 7

    V_idx, n = ext.get_idx_vertices(n)
    assert V_idx == [0, 1, 2, 3, 4, 5, 6]
    assert n == 7

    V_ext = ext.get_set_vertices_u_0(g)
    assert V_ext == [0, 1, 2, 3, 4, 5, 6, 7]

    V_ext = ext.get_set_vertices_u_0(n)
    assert V_ext == [0, 1, 2, 3, 4, 5, 6, 7]

    v_list = [1, 2]
    v_left = ext.get_v_left(g, v_list)
    v_left2 = ext.get_v_left(n, v_list)

    assert v_left == [3, 4, 5, 6, 7]
    assert v_left2 == [3, 4, 5, 6, 7]


def test_get_sets():
    g = ext.get_graph_00()
    deadline = 3
    list_input = [5, 2]

    S, V, T_ext = ext.get_sets_only(g, list_input, deadline)
    _, _, T = ext.get_sets_only(g, list_input, deadline, False)
    assert S == [1, 2]
    assert V == [1, 2, 3, 4, 5, 6, 7]
    assert T == [1, 2, 3]
    assert T_ext == [0, 1, 2, 3]

    S, V, T_ext, m, n = ext.get_sets_and_ranges(g, list_input, deadline)
    S, V, T, m, n = ext.get_sets_and_ranges(g, list_input, deadline, False)
    assert S == [1, 2]
    assert V == [1, 2, 3, 4, 5, 6, 7]
    assert T == [1, 2, 3]
    assert T_ext == [0, 1, 2, 3]
    assert n == 7
    assert m == 2


def test_get_graph():

    g0 = ext.get_graph_00()
    g1 = ext.get_graph_01()
    g2 = ext.get_graph_02()
    g3 = ext.get_graph_03()
    g4 = ext.get_graph_04()
    g5 = ext.get_graph_05()
    g6 = ext.get_graph_06()
    g7 = ext.get_graph_07()

    assert g0['name'] == 'G7V_test'
    assert g1['name'] == 'G60V'
    assert g2['name'] == 'G100V_grid'
    assert g3['name'] == 'G256V_grid'
    assert g4['name'] == 'G9V_grid'
    assert g5['name'] == 'G20_home'
    assert g6['name'] == 'G25_home'
    assert g7['name'] == 'G70V'

    V0, n0 = ext.get_set_vertices(g0)
    V1, n1 = ext.get_set_vertices(g1)
    V2, n2 = ext.get_set_vertices(g2)
    V3, n3 = ext.get_set_vertices(g3)
    V4, n4 = ext.get_set_vertices(g4)
    V5, n5 = ext.get_set_vertices(g5)
    V6, n6 = ext.get_set_vertices(g6)
    V7, n7 = ext.get_set_vertices(g7)

    assert n0 == 7
    assert n1 == 60
    assert n2 == 100
    assert n3 == 256
    assert n4 == 9
    assert n5 == 20
    assert n6 == 25
    assert n7 == 70

    g0 = ext.get_graph('G7V_test')
    g1 = ext.get_graph('G60V')
    g2 = ext.get_graph('G100V_grid')
    g3 = ext.get_graph('G256V_grid')
    g4 = ext.get_graph('G9V_grid')
    g5 = ext.get_graph('G20_home')
    g6 = ext.get_graph('G25_home')
    g7 = ext.get_graph('G70V')

    assert g0['name'] == 'G7V_test'
    assert g1['name'] == 'G60V'
    assert g2['name'] == 'G100V_grid'
    assert g3['name'] == 'G256V_grid'
    assert g4['name'] == 'G9V_grid'
    assert g5['name'] == 'G20_home'
    assert g6['name'] == 'G25_home'
    assert g7['name'] == 'G70V'

    V0, n0 = ext.get_set_vertices(g0)
    V1, n1 = ext.get_set_vertices(g1)
    V2, n2 = ext.get_set_vertices(g2)
    V3, n3 = ext.get_set_vertices(g3)
    V4, n4 = ext.get_set_vertices(g4)
    V5, n5 = ext.get_set_vertices(g5)
    V6, n6 = ext.get_set_vertices(g6)
    V7, n7 = ext.get_set_vertices(g7)

    assert n0 == 7
    assert n1 == 60
    assert n2 == 100
    assert n3 == 256
    assert n4 == 9
    assert n5 == 20
    assert n6 == 25
    assert n7 == 70








