
from core import extract_info as ext
from core import create_parameters as cp

from classes.inputs import MyInputs
from classes.belief import MyBelief


def test_belief_vector():
    """Test b_0 for vertices with equal probability"""
    # load graph
    graph_file = 'G7V_test.p'
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
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)

    v0_target = [1, 5, 7]
    init_prob = [0.2, 0.5, 0.3]
    b_0, M = cp.my_target_motion(g, v0_target, init_prob)
    assert b_0 == [0, 0.2, 0, 0, 0, 0.5, 0, 0.3]


def test_markovian_matrix():
    """Test Markovian matrix M for random motion"""
    # load graph
    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)

    v0_target = [1, 5, 7]
    init_prob = [0.2, 0.5, 0.3]
    b_0, M = cp.my_target_motion(g, v0_target, init_prob)

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
    graph_file = 'G64V_grid'
    g = ext.get_graph(graph_file)

    v_target = [1, 2, 3]
    v_searchers = [5]
    target_motion = 'random'
    distribution_type = 'uniform'
    capture_range = 1
    zeta = None

    b_0 = cp.set_initial_belief(g, v_target, distribution_type)
    M = cp.set_motion_matrix(g, target_motion)

    assert b_0[0] == 0.0
    assert b_0[1] == 1/3
    assert b_0[2] == 1/3
    assert b_0[3] == 1/3

    assert M[0][0] == 1/3
    assert M[-1][-1] == 1/3

    searchers = cp.create_dict_searchers(g, v_searchers, capture_range, zeta)

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


def test_draw_v_random():
    g = ext.get_graph_02()
    q = 4
    n = len(g.vs)

    v_list, v_left = cp.draw_v_random(g, q)

    assert isinstance(v_list, list)
    assert isinstance(v_left, list)
    assert len(v_list) == 4
    assert len(v_left) == (n - q)


def test_set_initial_belief():
    g = ext.get_graph_00()
    n = len(g.vs)
    v0_target = [7]
    belief_distribution = "uniform"

    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)

    assert len(b_0) == (n + 1)
    for i in range(n):
        assert b_0[i] == 0.0
    assert b_0[n] == 1

    v_list = cp.v_list_from_belief(b_0)

    assert len(v_list) == len(v0_target)
    assert v_list[0] == v0_target[0]


def test_placement_list():
    v_list = [7]
    v_s = [1, 2]

    specs = my_specs()

    assert v_s == specs.start_searcher_v
    assert v_list == specs.start_target_v_list

    v_list1 = cp.placement_list(specs, 't')
    assert isinstance(v_list1, list)
    check = cp.check_same_vertex(v_list1, v_s)
    assert check is True

    v_s1 = cp.placement_list(specs, 's')
    assert isinstance(v_s1, list)
    check = cp.check_same_vertex(v_list, v_s1)
    assert check is True


def test_create_belief():
    g = ext.get_graph_00()
    n = len(g.vs)
    v0_target = [7]
    b_0_true = [0, 0, 0, 0, 0, 0, 0, 1]
    belief_distribution = "uniform"

    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    v_list = cp.v_list_from_belief(b_0)
    assert v_list == v0_target
    assert b_0 == b_0_true

    belief1 = MyBelief(b_0)
    assert belief1.stored[0] == b_0
    assert belief1.milp_init_belief == b_0
    assert belief1.new == b_0
    assert belief1.start_belief == b_0

    specs = my_specs()
    assert specs.start_target_v_list == v0_target
    assert specs.b0 is None
    assert specs.belief_distribution == "uniform"

    v_list2 = cp.placement_list(specs, 't')
    assert v_list == v_list2
    assert v0_target == v_list2

    b_02 = cp.set_initial_belief(g, v_list2, belief_distribution)
    assert b_02 == b_0
    assert b_02 == b_0_true

    belief2 = cp.create_belief(specs)
    assert belief2.stored[0] == b_0
    assert belief2.milp_init_belief == b_0
    assert belief2.new == b_0
    assert belief2.start_belief == b_0


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

    v_target2 = [1, 2]
    init_is_ok = cp.check_reachability(g, capture_range, v_target2, v_searchers)
    init_is_ok1 = cp.check_reachability(g, capture_range, v_target2, v_searchers1)

    assert init_is_ok is False
    assert init_is_ok1 is True


def my_specs():
    theta = 2
    deadline = 6
    horizon = 3
    solver_type = 'central'

    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    target_motion = 'random'
    belief_distribution = 'uniform'

    v0_target = [7]
    v0_searchers = [1, 2]

    specs = MyInputs()

    specs.set_graph(0)
    specs.set_theta(theta)
    specs.set_deadline(deadline)
    specs.set_solver_type(solver_type)
    specs.set_horizon(horizon)
    specs.set_start_target(v0_target)
    specs.set_start_searchers(v0_searchers)
    specs.set_target_motion(target_motion)
    specs.set_belief_distribution(belief_distribution)

    return specs

