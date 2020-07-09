from core import plan_fun as pln
from core import construct_model as cm
from core import create_parameters as cp
from core import extract_info as ext
from core import milp_fun as mf
from core import sim_fun as sf
from classes.inputs import MyInputs
from gurobipy import *


# TODO fix these two things
def parameters_sim():

    today_run = 0
    gamma = 0.99
    theta = 2
    deadline = 6
    horizon = 3
    solver_type = 'central'

    graph_file = 'G7V7E.p'
    g = ext.get_graph(graph_file)
    target_motion = 'random'
    belief_distribution = 'uniform'

    v0_target = [7]
    v0_searchers = [1, 2]

    return theta, deadline, solver_type, horizon, g, v0_target, v0_searchers, target_motion, belief_distribution


def get_solver_param(sim_param=1):
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


def my_specs():
    theta, deadline, solver_type, horizon, g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    specs = MyInputs()

    specs.set_theta(theta)
    specs.set_deadline(deadline)
    specs.set_solver_type(solver_type)
    specs.set_horizon(horizon)
    specs.set_graph(0)
    specs.set_start_target_list(v0_target)
    specs.set_start_searchers(v0_searchers)
    specs.set_target_motion(target_motion)
    specs.set_belief_distribution(belief_distribution)

    return specs


# ---------------------------------------
def test_inputs():
    theta, deadline, solver_type, horizon, g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    specs = MyInputs()

    specs.set_theta(theta)
    specs.set_deadline(deadline)
    specs.set_solver_type(solver_type)
    specs.set_horizon(horizon)
    specs.set_graph(0)
    specs.set_start_target_list(v0_target)
    specs.set_start_searchers(v0_searchers)
    specs.set_target_motion(target_motion)
    specs.set_belief_distribution(belief_distribution)

    assert g["name"] == specs.graph["name"]
    assert specs.horizon == horizon
    assert specs.theta == theta
    assert specs.start_target_random is False
    assert specs.start_target_true == v0_target[0]
    assert specs.start_searcher_random is False
    assert specs.start_searcher_v == v0_searchers
    assert specs.target_motion == target_motion
    assert specs.belief_distribution == belief_distribution


def test_init_wrapper():
    # TODO check
    # belief = cp.create_belief(specs) > v_list = placement_list(specs, 't') >
    # out_of_reach = check_reachability(g, specs.capture_range, v_list, v_taken) >
    # distance = ext.get_node_distance(g, vt, v) > distance = spl[v1_idx][v2_idx]
    # ERROR list indices must be integers or slices, not list


    horizon = parameters_sim()[3]
    deadline = parameters_sim()[1]
    g = parameters_sim()[4]
    v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()[5:9]

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.my_motion_matrix(g, target_motion)
    searchers_ = cp.create_my_searchers(g, v0_searchers)
    # ________________________________________________________________________________________________________________

    specs = my_specs()
    # initialize instances of classes
    belief, searchers, solver_data, target = pln.init_wrapper(specs)

    assert belief.stored[0] == b_0
    assert belief.milp_init_belief == b_0
    assert belief.new == b_0
    assert belief.start_belief == b_0

    assert target.start_possible == v0_target
    assert target.start_true == target.stored_v_true[0]
    assert target.motion_matrix == M
    assert target.stored_v_true[0] in set(v0_target)
    assert target.stored_v_possible[0] == v0_target

    counter_s = 1
    for s_id in searchers.keys():
        idx = s_id - 1
        s = searchers[s_id]
        assert s.id == s_id
        assert s.start == v0_searchers[idx]
        assert s.start in set(v0_searchers)
        assert all(s.capture_matrices) == all(searchers_[s_id].capture_matrices)
        assert len(s.path_planned) == 0
        assert s.path_taken[0] == searchers_[s_id].start
        counter_s = counter_s + 1

    assert solver_data.solver_type == 'central'
    assert solver_data.theta == 2
    assert solver_data.horizon[0] == horizon
    assert solver_data.deadline == deadline


def test_update_start_searchers():

    # initial
    horizon = parameters_sim()[3]
    g = parameters_sim()[4]
    v0_searchers = parameters_sim()[5]
    searchers = cp.create_my_searchers(g, v0_searchers)

    # fake position
    fake_pos = dict()
    fake_pos[1] = 10
    fake_pos[2] = 11

    # update searcher position
    searchers = pln.searchers_next_position(searchers, fake_pos)

    pos_list = ext.get_position_list(searchers)
    for s_id in searchers.keys():
        assert pos_list[s_id - 1] == fake_pos[s_id]
        assert searchers[s_id].current_pos == fake_pos[s_id]


def test_check_false_negative():

    deadline = parameters_sim()[1]
    g = parameters_sim()[4]
    v0_searchers = parameters_sim()[6]

    # no false negatives
    searchers = cp.create_my_searchers(g, v0_searchers)
    false_neg = cm.check_false_negatives(searchers)[0]

    # ________________________________________________________________________________________________________________

    # initialize parameters according to inputs
    capture_range = 0
    zeta = 0.2
    searchers_2 = cp.create_my_searchers(g, v0_searchers, capture_range, zeta)
    false_neg_2, zeta2 = cm.check_false_negatives(searchers_2)

    assert false_neg is False
    assert false_neg_2 is True
    assert zeta2 == zeta


# --------------------------------------------------------------------------------------


def test_run_solver_get_model_data():

    horizon = parameters_sim()[3]
    deadline = parameters_sim()[1]
    g = parameters_sim()[4]
    v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()[5:9]

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.my_motion_matrix(g, target_motion)
    searchers_ = cp.create_my_searchers(g, v0_searchers)

    # solve: 1
    results, md1 = mf.run_gurobi(g, horizon, searchers_info, b_0, M, 0.99)
    x, b = mf.query_variables(md1)
    obj_fun1, time_sol1, gap1, threads1 = mf.get_model_data(md1)

    # solve: 2
    obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, searchers_info, b_0, M)

    # solve: 3
    specs = my_specs()
    # initialize instances of classes
    path = pln.run_default_planner(specs)

    assert x == x_searchers
    assert b == b_target

    assert obj_fun == md1.objVal
    assert round(time_sol, 2) == round(md1.Runtime, 2)
    assert gap == md1.MIPGap

    assert obj_fun == obj_fun1
    assert round(time_sol1, 4) == round(md1.Runtime, 4)
    assert gap == gap1

    assert threads == threads1


def test_get_positions_searchers():

    # GET parameters for the simulation, according to sim_param
    horizon, theta, deadline, solver_type = pmi.get_solver_param()

    # GET parameters for the MILP solver
    # get horizon, graph etc according to inputs_opt parameter (pre-made inputs)
    g, v0_target, v0_searchers, target_motion, belief_distribution = pmi.get_inputs()

    # get sets for easy iteration
    S, V, n, Tau = ext.get_sets_only(v0_searchers, deadline, g)

    # ________________________________________________________________________________________________________________

    # INITIALIZE

    # initialize parameters according to inputs
    b_0, M, s_info = cp.init_parameters(g, v0_target, v0_searchers, target_motion, belief_distribution)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, sim_data = sf.init_all_classes(horizon, deadline, theta, g, 'central', b_0, s_info,
                                                   v0_target, M)

    searchers = cp.create_my_searchers(g, v0_searchers)

    obj_fun, time_sol, gap, x_searchers, b_target, threads = core.plan_fun.run_solver(g, horizon, s_info, b_0, M)

    searchers, s_pos = core.plan_fun.xs_to_path(x_searchers, V, Tau, searchers)

    assert s_pos[1, 0] == 1
    assert s_pos[1, 1] == 3
    assert s_pos[1, 2] == 5
    assert s_pos[1, 3] == 6

    assert s_pos[2, 0] == 2
    assert s_pos[2, 1] == 5
    assert s_pos[2, 2] == 6
    assert s_pos[2, 3] == 7

    assert searchers[1].path_planned[0] == [1, 3, 5, 6]
    assert searchers[2].path_planned[0] == [2, 5, 6, 7]

    new_pos = core.plan_fun.next_from_path(searchers, s_pos, 1)
    searchers = core.plan_fun.searchers_next_position(searchers, new_pos)

    assert searchers[1].path_taken[1] == 3
    assert searchers[2].path_taken[1] == 5

    assert searchers[1].current_pos == 3
    assert searchers[2].current_pos == 5

    # get next time and vertex (after evolving position)
    next_time, v_target = ext.get_last_info(target.stored_v_true)

    # evolve searcher position
    searchers[1].current_pos = v_target

    searchers, target = sf.check_for_capture(searchers, target)

    assert target.is_captured is True


def test_time_consistency():
    # GET parameters for the simulation, according to sim_param
    horizon, theta, deadline, solver_type = pmi.get_solver_param()

    # GET parameters for the MILP solver
    # get horizon, graph etc according to inputs_opt parameter (pre-made inputs)
    g, v0_target, v0_searchers, target_motion, belief_distribution = pmi.get_inputs()

    # get sets for easy iteration
    S, V, n, Tau = ext.get_sets_only(v0_searchers, deadline, g)

    # ________________________________________________________________________________________________________________

    # INITIALIZE

    # initialize parameters according to inputs
    b_0, M, s_info = cp.init_parameters(g, v0_target, v0_searchers, target_motion, belief_distribution)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, sim_data = sf.init_all_classes(horizon, deadline, theta, g, solver_type, b_0,  s_info,
                                                   v0_target, M)

    searchers = cp.create_my_searchers(g, v0_searchers)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    s_pos_plan = {}
    new_pos = {}

    # FIRST ITERATION

    # call for model solver wrapper according to centralized or decentralized solver and return the solver data
    obj_fun, time_sol, gap, x_searchers, b_target, threads = core.plan_fun.run_solver(g, horizon, s_info, belief.new, M, 1.5,
                                                                                      solver_type)
    # save the new data
    sim_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, horizon)

    # get position of each searcher at each time-step based on x[s][v, t] variable
    searchers, s_pos_plan = core.plan_fun.xs_to_path(x_searchers, V, Tau, searchers)

    # reset time-steps of planning
    t_plan = 1

    # EVOLVE THINGS
    new_pos = core.plan_fun.next_from_path(searchers, s_pos_plan, t_plan)

    # evolve searcher position
    searchers = core.plan_fun.searchers_next_position(searchers, new_pos)

    # update belief
    belief.update(s_info, new_pos, M, n)

    # update target
    target = sf.evolve_target(target, belief.new)

    # next time-step
    t, t_plan = t + 1, t_plan + 1

    assert t == 1
    assert t_plan == 2

    # get next time and vertex (after evolving position)
    t1, v_t = ext.get_last_info(target.stored_v_true)
    t2, v_s = ext.get_last_info(searchers[1].path_taken)

    assert t1 == t2
    assert t1 == t




