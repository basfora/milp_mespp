from core import plan_fun as pln
from core import construct_model as cm
from core import create_parameters as cp
from core import extract_info as ext
from core import milp_fun as mf
from core import sim_fun as sf
from classes.inputs import MyInputs
from classes.solver_data import MySolverData
from classes.belief import MyBelief
from classes.target import MyTarget
from gurobipy import *


def parameters_sim():

    graph_file = 'G7V_test.p'
    g = ext.get_graph(graph_file)
    target_motion = 'random'
    belief_distribution = 'uniform'

    v0_target = [7]
    v0_searchers = [1, 2]

    return g, v0_target, v0_searchers, target_motion, belief_distribution


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
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    specs = MyInputs()

    specs.set_graph(0)
    specs.set_theta(theta)
    specs.set_deadline(deadline)
    specs.set_solver_type(solver_type)
    specs.set_horizon(horizon)
    specs.set_start_target_list(v0_target)
    specs.set_start_searchers(v0_searchers)
    specs.set_target_motion(target_motion)
    specs.set_belief_distribution(belief_distribution)

    return specs


# ---------------------------------------
def test_inputs():
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

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
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    assert v0_target == [7]

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.set_motion_matrix(g, target_motion)
    searchers_ = cp.create_dict_searchers(g, v0_searchers)
    # ________________________________________________________________________________________________________________

    specs = my_specs()
    # initialize instances of classes
    belief, searchers, solver_data, target = pln.init_wrapper(specs)

    belief1 = cp.create_belief(specs)

    assert belief1.stored[0] == b_0
    assert belief1.milp_init_belief == b_0
    assert belief1.new == b_0
    assert belief1.start_belief == b_0

    assert belief.stored[0] == b_0
    assert belief.milp_init_belief == b_0
    assert belief.new == b_0
    assert belief.start_belief == b_0

    assert target.start_possible == v0_target
    assert target.start_true == target.stored_v_true[0]
    assert target.motion_matrix == M
    assert target.stored_v_true[0] in set(v0_target)
    assert target.stored_v_possible[0] == v0_target

    assert specs.size_team == len(searchers.keys())
    for s_id in searchers.keys():
        idx = s_id - 1
        s = searchers[s_id]
        assert s.id == s_id
        assert s.start == v0_searchers[idx]
        assert s.start in set(v0_searchers)
        assert all(s.capture_matrices) == all(searchers_[s_id].capture_matrices)
        assert len(s.path_planned) == 0
        assert s.path_taken[0] == searchers_[s_id].start

    assert solver_data.solver_type == 'central'
    assert solver_data.theta == 2
    assert solver_data.horizon[0] == horizon
    assert solver_data.deadline == deadline


def test_update_start_searchers():

    # initial
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()
    searchers = cp.create_dict_searchers(g, v0_searchers)

    # fake position
    fake_pos = dict()
    fake_pos[1] = 10
    fake_pos[2] = 11

    # update searcher position
    searchers = pln.searchers_evolve(searchers, fake_pos)

    pos_list = ext.get_searchers_positions(searchers)
    for s_id in searchers.keys():
        assert pos_list[s_id - 1] == fake_pos[s_id]
        assert searchers[s_id].current_pos == fake_pos[s_id]


def test_check_false_negative():
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    # no false negatives
    searchers = cp.create_dict_searchers(g, v0_searchers)
    false_neg = cm.check_false_negatives(searchers)[0]

    # ________________________________________________________________________________________________________________

    # initialize parameters according to inputs
    capture_range = 0
    zeta = 0.2
    searchers_2 = cp.create_dict_searchers(g, v0_searchers, capture_range, zeta)
    false_neg_2, zeta2 = cm.check_false_negatives(searchers_2)

    assert false_neg is False
    assert false_neg_2 is True
    assert zeta2 == zeta


# --------------------------------------------------------------------------------------

def test_run_solver_get_model_data():
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()
    gamma = 0.99
    timeout = 60

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.set_motion_matrix(g, target_motion)
    searchers = cp.create_dict_searchers(g, v0_searchers)

    # solve: 1 [low level]
    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers)
    # create model
    md = mf.create_model()
    # add variables
    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, searchers)
    # add constraints (central algorithm)
    mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b_0, M)
    # objective function
    mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout)
    # update
    md.update()
    # Optimize model
    md.optimize()
    x_s1, b_target1 = mf.query_variables(md)
    obj_fun1, time_sol1, gap1, threads1 = mf.get_model_data(md)
    pi_dict1 = pln.xs_to_path(x_s1)
    path1 = pln.path_as_list(pi_dict1)

    # solve: 2
    obj_fun2, time_sol2, gap2, x_s2, b_target2, threads2 = pln.run_solver(g, horizon, searchers, b_0, M)
    pi_dict2 = pln.xs_to_path(x_s2)
    path2 = pln.path_as_list(pi_dict2)

    # solve: 3
    specs = my_specs()
    # initialize instances of classes
    path3 = pln.run_planner(specs)

    assert obj_fun1 == md.objVal
    assert round(time_sol1, 2) == round(md.Runtime, 2)
    assert gap1 == md.MIPGap

    # 1 x 2
    assert x_s1 == x_s2
    assert b_target1 == b_target2

    assert obj_fun2 == obj_fun1
    assert round(time_sol2, 2) == round(time_sol1, 2)
    assert gap2 == gap1
    assert threads2 == threads1

    # paths
    assert pi_dict1 == pi_dict2
    assert path1 == path2
    # 1 x 3
    assert path1 == path3


def test_get_positions_searchers():

    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()

    # ________________________________________________________________________________________________________________

    # INITIALIZE

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.set_motion_matrix(g, target_motion)
    searchers = cp.create_dict_searchers(g, v0_searchers)

    specs = my_specs()
    target = cp.create_target(specs)

    obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, searchers, b_0, M)

    # get position of each searcher at each time-step based on x[s, v, t] variable
    searchers, s_pos = pln.update_plan(searchers, x_searchers)

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

    new_pos = pln.next_from_path(s_pos, 1)
    searchers = pln.searchers_evolve(searchers, new_pos)

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
    horizon, theta, deadline, solver_type = get_solver_param()
    g, v0_target, v0_searchers, target_motion, belief_distribution = parameters_sim()
    gamma = 0.99

    # get sets for easy iteration
    V, n = ext.get_set_vertices(g)

    # ________________________________________________________________________________________________________________
    # INITIALIZE

    # initialize parameters according to inputs
    b_0 = cp.set_initial_belief(g, v0_target, belief_distribution)
    M = cp.set_motion_matrix(g, target_motion)
    searchers = cp.create_dict_searchers(g, v0_searchers)
    solver_data = MySolverData(horizon, deadline, theta, g, solver_type)
    belief = MyBelief(b_0)
    target = MyTarget(v0_target, M)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0

    # FIRST ITERATION
    # call for model solver wrapper according to centralized or decentralized solver and return the solver data
    obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, searchers, belief.new, M, gamma,
                                                                            solver_type)
    # save the new data
    solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, horizon)

    # get position of each searcher at each time-step based on x[s, v, t] variable
    searchers, path = pln.update_plan(searchers, x_searchers)

    # reset time-steps of planning
    t_plan = 1

    path_next_t = pln.next_from_path(path, t_plan)

    # evolve searcher position
    searchers = pln.searchers_evolve(searchers, path_next_t)

    # update belief
    belief.update(searchers, path_next_t, M, n)

    # update target
    target = sf.evolve_target(target, belief.new)

    # next time-step
    t, t_plan = t + 1, t_plan + 1

    assert t == 1
    assert t_plan == 2

    # get next time and vertex (after evolving position)
    t_t, v_t = ext.get_last_info(target.stored_v_true)
    assert target.current_pos == v_t
    t_s, v_s = ext.get_last_info(searchers[1].path_taken)

    assert t_t == t_s
    assert t_t == t

    # high level
    specs = my_specs()
    belief1, searchers1, solver_data1, target1 = pln.init_wrapper(specs)

    deadline1, horizon1, theta1, solver_type1, gamma1 = solver_data1.unpack()
    M1 = target1.unpack()

    assert deadline1 == deadline
    assert horizon1 == horizon
    assert theta1 == theta
    assert solver_type1 == solver_type
    assert gamma1 == gamma
    assert M1 == M

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t1, t_plan1 = 0, 0

    # FIRST ITERATION
    # call for model solver wrapper according to centralized or decentralized solver and return the solver data
    obj_fun1, time_sol1, gap1, x_searchers1, b_target1, threads1 = pln.run_solver(g, horizon1, searchers1, belief1.new,
                                                                                  M1, gamma1, solver_type1)

    assert obj_fun == obj_fun1
    assert round(time_sol, 2) == round(time_sol1, 2)
    assert gap == gap1
    assert x_searchers == x_searchers1
    assert b_target == b_target1
    assert threads == threads1

    # save the new data
    solver_data1.store_new_data(obj_fun1, time_sol1, gap1, threads1, x_searchers1, b_target1, horizon1)

    # get position of each searcher at each time-step based on x[s, v, t] variable
    searchers1, path1 = pln.update_plan(searchers1, x_searchers1)

    assert path == path1

    # reset time-steps of planning
    t_plan1 = 1

    path_next_t1 = pln.next_from_path(path1, t_plan1)

    assert path_next_t == path_next_t1

    # evolve searcher position
    searchers1 = pln.searchers_evolve(searchers1, path_next_t1)

    # update belief
    belief1.update(searchers1, path_next_t1, M1, n)

    # update target
    target1 = sf.evolve_target(target1, belief1.new)

    # next time-step
    t1, t_plan1 = t1 + 1, t_plan1 + 1

    assert t1 == 1
    assert t_plan1 == 2

    assert target1.start_possible == target.start_possible

    # get next time and vertex (after evolving position)
    t_t1, v_t1 = ext.get_last_info(target1.stored_v_true)
    t_s1, v_s1 = ext.get_last_info(searchers1[1].path_taken)

    assert t_t1 == t_s1
    assert t_t1 == t1

    assert t_t1 == t_t

    assert t_s1 == t_s
    assert v_s1 == v_s





















