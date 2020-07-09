from core import extract_info as ext
from core import create_parameters as cp


def test_belief_vector():
    """Test b_0 for vertices with equal probability"""
    # load graph
    graph_file = 'G7V7E.p'
    g = ext.get_graph(graph_file)
    v_list = [5]
    type_distribution = 'uniform'

    b_0 = cp.set_initial_belief(g, v_list, type_distribution)
    assert b_0 == [0, 0, 0, 0, 0, 1, 0, 0]

    v_list = [1, 7]
    b_0 = cp.set_initial_belief(g, v_list, type_distribution)
    assert b_0 == [0, 1/2, 0, 0, 0, 0, 0, 1/2]


def test_belief_vector_prob():
    """Test b_0 for several vertices, user defined probability"""
    # load graph
    graph_file = 'G7V7E.p'
    g = ext.get_graph(graph_file)

    v_target_init = [1, 5, 7]
    init_prob = [0.2, 0.5, 0.3]
    b_0, M = cp.my_target_motion(g, v_target_init, init_prob)
    assert b_0 == [0, 0.2, 0, 0, 0, 0.5, 0, 0.3]


def test_markovian_matrix():
    """Test Markovian matrix M for random motion"""
    # load graph
    graph_file = 'G7V7E.p'
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
    distribution_type = 'uniform'
    capture_range = 1
    zeta = None

    b_0 = cp.set_initial_belief(g, v_target, distribution_type)
    M = cp.my_motion_matrix(g, target_motion)

    assert b_0[0] == 0.0
    assert b_0[1] == 1/3
    assert b_0[2] == 1/3
    assert b_0[3] == 1/3

    assert M[0][0] == 1/3
    assert M[-1][-1] == 1/3

    searchers = cp.create_my_searchers(g, v_searchers, capture_range, zeta)

    s_id = 1
    u = 1

    s = searchers[s_id]
    C = s.get_capture_matrix(u)

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
