from core import extract_info as ext
from core import construct_model as cm
from core import create_parameters as cp
from core import aux_classes as ac
import numpy as np
from core.classes.class_belief import MyBelief
from core.classes.class_target import MyTarget
from core.classes.class_searcher import MySearcher
from core.classes.class_solverData import MySolverData
from core import analyze_results as ar
from gurobipy import *
from core import milp_model_functions as mf


def parameters_7v_random_motion():
    """Parameters pre-defined for unit tests"""
    # load graph
    graph_file = 'G_7V7E.p'
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
    b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution)
    n = 7
    return n, b_0, M, searchers_info


def parameters_7v_random_motion2():
    """Parameters pre-defined for unit tests"""
    # load graph
    graph_file = 'G_7V7E.p'
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
    b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution)
    n = 7
    return n, b_0, M, searchers_info, g


def test_belief_class_init():
    """Test class target"""

    n, b_0, M, searchers_info = parameters_7v_random_motion()

    b = MyBelief(b_0)

    dict_belief = b.stored

    assert b.start_belief == b_0
    assert dict_belief[0] == b_0
    assert b.milp_init_belief == b_0


def test_belief_class_update():
    n, b_0, M, searchers_info = parameters_7v_random_motion()

    # searchers position
    new_pos = dict()
    new_pos[1] = 1
    new_pos[2] = 2

    t = 0

    b = MyBelief(b_0)
    # class method, update belief
    b.update(searchers_info, new_pos, M, n)

    # find the product of the capture matrices, s = 1...m
    prod_C = ac.product_capture_matrix(searchers_info, new_pos, n)

    # assemble the matrix for multiplication
    big_M = ac.assemble_big_matrix(n, M)
    new_belief = np.array(b_0).dot(big_M).dot(prod_C)
    a = new_belief.tolist()

    # make sure no information was changed
    assert b.start_belief == b_0
    assert b.stored[0] == b_0
    assert b.milp_init_belief == b_0
    # make sure it was updated
    nb = b.stored[1]
    assert isinstance(nb, list)
    assert b.stored[1] == a


def test_belief_update_init():
    n, b_0, M, searchers_info = parameters_7v_random_motion()

    b = MyBelief(b_0)
    new_belief = [0, 0.5, 0, 0, 0, 0, 0.5, 0]

    # searchers position
    s_pos = {}
    s_pos[(1, 0)] = 1
    s_pos[(2, 0)] = 2

    b.new_init_belief(new_belief)

    assert b.milp_init_belief == new_belief


def test_target_class_init():

    v_target = [6, 7]
    v_target_true = 7

    n, b_0, M, searchers_info = parameters_7v_random_motion()

    target = MyTarget(v_target, M, v_target_true)

    # possible and true vertices
    assert target.start_possible == v_target
    assert target.start_true == v_target_true

    # motion matrix
    assert target.motion_matrix == M
    assert target.motion_matrix_true == M

    # stored initial
    assert target.stored_v_possible[0] == v_target
    assert target.stored_v_true[0] == v_target_true

    # status capture
    assert target.is_captured is False
    assert target.capture_time is None


def test_target_class_init2():

    v_target = [6, 7]

    n, b_0, M, searchers_info = parameters_7v_random_motion()

    bad_inited = False
    try:
        target = MyTarget(v_target, M)
    except:
        bad_inited = True

    assert bad_inited is True

    v_target = [7]
    target = MyTarget(v_target, M)

    assert target.start_true == 7


def test_target_update_status():
    v_target = [7]
    n, b_0, M, searchers_info = parameters_7v_random_motion()

    target = MyTarget(v_target, M)

    target.update_status(True)

    assert target.is_captured is True
    assert target.capture_time == 0


def test_target_evolve_position():
    v_target = [7]
    n, b_0, M, searchers_info = parameters_7v_random_motion()

    target = MyTarget(v_target, M)

    target.evolve_true_position()

    new_b = [0, 0, 0, 0, 0, 0, 0.5, 0.5]
    target.evolve_possible_position(new_b)

    assert target.stored_v_true[0] == 7
    if target.stored_v_true[1] == 7 or target.stored_v_true[1] == 6:
        evolution = True
    else:
        evolution = False
    assert evolution is True

    assert target.stored_v_possible[0] == [7]
    assert target.stored_v_possible[1] == [6, 7]


def test_searcher_class():

    n, b_0, M, searchers_info = parameters_7v_random_motion()

    s = MySearcher(1, searchers_info)

    assert s.id == 1
    assert s.start == 1
    assert s.capture_matrices == cm.get_all_capture_matrices(searchers_info, 1)
    assert s.init_milp == 1
    assert s.catcher is False
    assert len(s.path_planned) == 0
    assert len(s.path_taken) == 1
    assert s.path_taken[0] == 1

    path_planned1 = [1, 2, 3, 4]
    s.store_path_planned(path_planned1)

    assert s.path_planned[0] == path_planned1
    assert s.init_milp == 1

    next_position1 = 2
    next_position2 = 3

    s.evolve_position(next_position1)
    s.evolve_position(next_position2)

    assert s.path_taken[1] == next_position1
    assert s.path_taken[2] == next_position2

    path_planned2 = [3, 4, 5, 6]

    s.store_path_planned(path_planned2)

    assert s.path_planned[2] == path_planned2
    assert s.init_milp == 3


def test_solver_data_class():

    horizon = 3
    n, b_0, M, searchers_info, g = parameters_7v_random_motion2()
    # solve
    # create model
    md = Model("my_model")

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers_info)

    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, )

    mf.add_constraints(md, g, my_vars, searchers_info, vertices_t, horizon, b_0, M)

    mf.set_solver_parameters(md, 1.5, horizon, my_vars)

    md.update()
    # Optimize model
    md.optimize()

    s_pos, b_target = ar.query_variables(md, searchers_info)

    obj_fun = md.objVal
    gap = md.MIPGap
    time_sol = round(md.Runtime, 4)
    threads = md.Params.Threads
    deadline = 6
    theta = 3

    my_data = MySolverData(horizon, deadline, theta, g, 'central')
    t = 0
    my_data.store_new_data(obj_fun, time_sol, gap, threads, s_pos, b_target, t, horizon)

    assert my_data.obj_value[0] == obj_fun
    assert my_data.solve_time[0] == time_sol
    assert my_data.threads[0] == md.Params.Threads
    assert my_data.gap[0] == gap
    assert my_data.s_pos[0] == s_pos
    assert my_data.belief[0] == b_target
    assert my_data.solver_type == 'central'
    assert my_data.threads[0] == threads

