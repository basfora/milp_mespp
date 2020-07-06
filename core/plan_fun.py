"""Planner functions
- central wrapper
- distributed wrapper
- run_solver
"""

import numpy as np
from core import extract_info as ext, milp_fun as mf, construct_model as cm
from core import create_parameters as cp
from core import sim_fun as sf

from core import retrieve_data as rd
from gurobipy import *


# main wrappers
def run_solver(g, horizon, searchers, b0, M_target, gamma=0.99, opt='central', timeout=30 * 60, n_inter=1, pre_solve=-1):
    """Run solver according to type of planning specified"""

    if opt == 'central':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = central_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout)

    elif opt == 'distributed':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = distributed_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout, n_inter, pre_solve)
    else:
        obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.none_model_vars()

    return obj_fun, time_sol, gap, x_searchers, b_target, threads


def central_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout):
    """Add variables, constraints, objective function and solve the model
    compute all paths"""

    solver_type = 'central'

    # OK with searchers
    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers)

    # create model
    md = mf.create_model()

    # add variables
    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, searchers)

    # add constraints (central algorithm)
    mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b0, M_target)

    # objective function
    mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout)

    # solve and save results
    obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.solve_model(md)

    # clean things
    md.reset()
    md.terminate()
    del md
    #

    # clean things
    return obj_fun, time_sol, gap, x_searchers, b_target, threads


def distributed_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout=5, n_inter=1, pre_solver=-1):
    """Distributed version of the algorithm """

    # parameter to stop iterations
    # number of full loops s= 1..m
    n_it = n_inter

    # iterative parameters
    total_time_sol = 0
    previous_obj_fun = 0
    my_counter = 0

    # temporary path for the searchers
    temp_pi = init_temp_path(searchers, horizon)

    # get last searcher number [m]
    m = ext.get_last_info(searchers)[0]

    obj_fun_list = {}
    time_sol_list = {}
    gap_list = {}

    while True:

        for s_id in searchers.keys():
            # create model
            md = mf.create_model()

            temp_pi['current_searcher'] = s_id

            start, vertices_t, times_v = cm.get_vertices_and_steps_distributed(g, horizon, searchers, temp_pi)

            # add variables
            my_vars = mf.add_variables(md, g, horizon, start, vertices_t, searchers)

            mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b0, M_target)

            # objective function
            mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout, pre_solver)

            # solve and save results
            obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.solve_model(md)

            if md.SolCount == 0:
                # problem was infeasible or other error (no solution found)
                print('Error, no solution found!')
                obj_fun, time_sol, gap, threads = -1, -1, -1, -1
                # keep previous belief
                b_target = {}
                v = 0
                for el in b0:
                    b_target[(v, 0)] = el
                    v += 1

                x_searchers = keep_all_still(temp_pi)

            # clean things
            md.reset()
            md.terminate()
            del md

            # ------------------------------------------------------
            # append to the list
            obj_fun_list[s_id] = obj_fun
            time_sol_list[s_id] = time_sol
            gap_list[s_id] = gap

            # save the current searcher's path
            temp_pi = update_temp_path(x_searchers, temp_pi, s_id)

            total_time_sol = total_time_sol + time_sol_list[s_id]

            # end of a loop through searchers
            if s_id == m:
                # compute difference between previous and current objective functions
                delta_obj = abs(previous_obj_fun - obj_fun)

                # iterate
                previous_obj_fun = obj_fun
                my_counter = my_counter + 1

                # check for stoppers
                # either the objective function converged or iterated as much as I wanted
                if (delta_obj < 1e-4) or (my_counter >= n_it):
                    time_sol_list['total'] = total_time_sol
                    # clean and delete

                    disposeDefaultEnv()

                    return obj_fun_list, time_sol_list, gap_list, x_searchers, b_target, threads


def plan_init_wrapper(exp_inputs):

    solver_data = cp.create_solver_data(exp_inputs)

    g = exp_inputs.graph

    capture_range = exp_inputs.capture_range
    m = exp_inputs.size_team
    zeta = exp_inputs.zeta

    # belief stuff
    belief_distribution = exp_inputs.belief_distribution

    if exp_inputs.start_searcher_random:
        # gather seeds
        s_seed = exp_inputs.searcher_seed
        t_seed = exp_inputs.target_seed

        my_seed = dict()
        my_seed['searcher'] = s_seed
        my_seed['target'] = t_seed

        # target
        t_possible_nodes = exp_inputs.qty_possible_nodes

        # searchers
        init_is_ok = False
        v_target, v_searchers = None, None

        while init_is_ok is False:

            v_searchers, v_target = cp.random_init_pos(g, m, t_possible_nodes, my_seed)

            init_is_ok = cp.check_reachability(g, capture_range, v_target, v_searchers)

            if init_is_ok is False:
                print('Target within range --> target: ' + str(v_target) + 'searcher ' + str(v_searchers))
                my_seed['searcher'] = my_seed['searcher'] + 500

            # initialize parameters if everything is ok
            b0, M, s_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution,
                                               capture_range, zeta)
    else:
        v_searchers = exp_inputs.start_searcher_v

    # belief
    belief = MyBelief(init_belief)

    searchers = cp.create_my_searchers(g, v_searchers, capture_range, zeta)

    print('Start target: %d, searcher: %d ' % (target.current_pos, searchers[1].start))

    return belief, searchers, solver_data



# ----------------------------------------------------
# searchers
# define searchers path
def keep_all_still(temp_pi):
    """Return the variable x_s correspondent the current searchers' positions
    input: temp_pi(s, t) = v
    output: x_s(s, v, t) = 1"""

    x_searchers = path_to_xs(temp_pi)

    print('Keeping searchers still.')

    return x_searchers


def init_temp_path(searchers: dict, horizon: int):
    """If no path was computed yet, assume all searchers will stay at the start position
    :param searchers: dictionary of searcher class
    :param horizon: planning horizon (h)"""

    Tau = ext.get_set_ext_time(horizon)
    start = ext.get_start_set(searchers)
    # S_ and Tau
    S, m = ext.get_set_searchers(start)

    temp_pi = dict()
    temp_pi['current_searcher'] = None

    for s in S:
        s_idx = ext.get_python_idx(s)
        v = start[s_idx]
        for t in Tau:
            temp_pi[s, t] = v

    return temp_pi


def update_temp_path(searchers: dict, temp_pi: dict, my_s: int):
    """Integrate computed path of a single searcher in the temporary path of all searchers
    :param searchers: dictionary of searchers (each is a searcher class)
    :param temp_pi (s, t) = v
    :param my_s : 1, 2...m
    """

    for k in searchers.keys():
        s, v, t = ext.get_from_tuple_key(k)
        if s == my_s and searchers.get(k) == 1:
            temp_pi[(my_s, t)] = v

    return temp_pi


# get searchers path after planner
def xs_to_path(x_s: dict):
    """Get x variables which are one and save it as the planned path path[s_id, time]: v
    save planned path in searchers
    Convert from
    x_s(s, v, t) = 1
    to
    OUTPUT pi(s, t) = v"""

    pi = dict()

    for k in x_s.keys():
        value = x_s.get(k)
        s, v, t = ext.get_from_tuple_key(k)

        if value == 1:
            pi[s, t] = v

    return pi


def path_to_xs(path: dict):
    """Convert from
    pi(s, t) = v
    to
    OUTPUT x_s(s, v, t) = 1"""

    x_searchers = {}
    for k in path.keys():

        # ignore first one (if it's temp_pi)
        if k == 'current_searcher':
            continue

        s, t = ext.get_from_tuple_key(k)
        # get vertex searcher is currently in
        v = path.get((s, t))

        x_searchers[(s, v, t)] = 1

    return x_searchers


def path_as_lists(path: dict):
    """Get sequence of vertices from path[s, t] = v
    for searcher s
    return as list [v0, v1, v2...vh]"""

    pi = dict()

    h = ext.get_h(path)
    m = ext.get_m(path)
    T = ext.get_set_ext_time(h)
    S = ext.get_set_searchers(m)[0]

    # loop through time
    for s in S:
        pi[s] = []
        for t in T:
            v = path[(s, t)]
            pi[s].append(v)

    return pi


def path_of_s(path: dict, s_id):
    """Get sequence of vertices [list] for searcher s_id
    INPUT path[s, t] = v
    OUTPUT [v0, ....v]"""

    pi = path_as_lists(path)

    return pi[s_id]


def next_from_path(path: dict, t_plan: int):
    """ get new position of searchers as new_pos = {s: v}"""

    m = ext.get_m(path)
    S = ext.get_set_searchers(m)[0]

    new_pos = dict()
    for s_id in S:
        new_pos[s_id] = path[(s_id, t_plan)]

    return new_pos


def get_all_from_xs(x_s):
    """Return list of (s, v, t, value) tuples from x_s"""

    my_list = []
    for k in x_s.keys():
        s, v, t = ext.get_from_tuple_key(k)
        value = x_s.get(k)
        my_list.append((s, v, t, value))

    return my_list


# modify searchers class
def update_plan(searchers: dict, x_s: dict):
    """Get new plan from x_searchers variable
    Store new plan on searchers class"""

    # get position of all searchers based on x[s, v, t] variable from solver
    path = xs_to_path(x_s)

    searchers = store_path(searchers, path)

    return searchers, path


def store_path(searchers: dict, path: dict):
    """
    Store paths for all searchers
    :param searchers [s] dict of searcher class
    :param path [s, t] = v
    """

    for s_id in searchers.keys():
        s = searchers[s_id]
        path_s = path_of_s(path, s_id)
        s.store_path_planned(path_s)

    return searchers


def searchers_next_position(searchers, new_pos):
    """call to evolve searchers position """

    for s_id in searchers.keys():
        searchers[s_id].evolve_position(new_pos[s_id])

    return searchers


# ----------------------------------------------------------------------------
def initialize_planner(my_graph, my_h):
    """Initialize the planner the pre-set parameters
    If needed, change parameters here"""

    # ---------------------------------------
    # fixed parameters
    m = 3
    turns = 200
    solver_type = 'distributed'
    # time stuff
    horizon = my_h
    deadline = horizon
    theta = horizon
    # ---------------------------------------
    # initialize default inputs
    exp_inputs = MyInputs()

    # graph number -- G25_home = 6
    exp_inputs.get_graph(my_graph)

    # solver parameter: central x distributed
    exp_inputs.set_solver_type(solver_type)

    # searchers' detection: capture range and false negatives
    exp_inputs.set_size_team(m)

    if my_graph == 2:
        exp_inputs.set_capture_range(1)
        exp_inputs.set_zeta(0.3)

    # time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
    exp_inputs.set_all_times(horizon)

    # repetitions for each configuration
    exp_inputs.set_number_of_runs(turns)

    exp_inputs.set_timeout(120)

    # target stuff
    exp_inputs.set_target_motion('random')

    return exp_inputs


def run_planner(specs):

    t = 0
    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(specs.graph, specs.size_team, specs.horizon)

    belief, target, searchers, solver_data = sf.my_init_wrapper(specs)

    M = target.unpack()
    timeout = specs.timeout

    # ------------------------------------------
    # call for model solver wrapper according to centralized or decentralized solver and return the solver data
    obj_fun, time_sol, gap, x_searchers, b_target, threads = run_solver(specs.graph, specs.horizon, searchers,
                                                                        belief.new, M, specs.gamma,
                                                                        specs.solver_type, timeout)

    # save the new data
    solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

    # break here if the problem was infeasible
    if time_sol is None or gap is None or obj_fun is None:
        # belief, target, searchers, solver_data
        return None

    # get position of each searcher at each time-step based on x[s][v, t] variable
    searchers, path = update_plan(searchers, x_searchers)

    return belief, target, searchers, solver_data


def list_n_nodes():

    turns = 200
    list_qty = []

    for i in range(0, turns):
        random_n = np.random.randint(low=2, high=15)
        list_qty.append(random_n)

    return list_qty


def this_folder(name_folder, parent_folder='data'):

    f_path = ext.get_whole_path(name_folder, parent_folder)

    return f_path


def create_txt(name_folder):

    my_path = this_folder(name_folder, 'data_plan')

    # get data
    data = rd.load_data(my_path)

    if data is None:
        print('X - ', sep=' ', end='', flush=True)

    # classes
    belief, target, searchers, solver_data, exp_inputs = rd.get_classes(data)

    rd.organize_data_make_files(belief, target, searchers, solver_data, exp_inputs, my_path)


# -----------------------------------------------------



