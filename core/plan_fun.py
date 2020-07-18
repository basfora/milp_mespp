"""Planner functions
- central wrapper
- distributed wrapper
- run_solver
"""

from core import extract_info as ext, milp_fun as mf, construct_model as cm
from core import create_parameters as cp
from gurobipy import *


def run_planner(specs=None, printout=True):
    """Initialize the planner the pre-set parameters
        Return path of searchers as list"""

    if specs is None:
        specs = cp.default_specs()

    belief, searchers, solver_data, target = init_wrapper(specs)

    # unpack parameters
    g = specs.graph
    h = specs.horizon
    b0 = belief.new
    M = target.unpack()

    obj_fun, time_sol, gap, x_s, b_target, threads = run_solver(g, h, searchers, b0, M)
    searchers, path_dict = update_plan(searchers, x_s)

    path_list = path_as_list(path_dict)

    if printout:
        print_path(x_s)

    return path_list


def run_solver(g, horizon, searchers, b0, M_target, gamma=0.99, solver_type='central', timeout=30 * 60, n_inter=1, pre_solve=-1):
    """Run solver according to type of planning specified"""

    if solver_type == 'central':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = central_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout)

    elif solver_type == 'distributed':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = distributed_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout, n_inter, pre_solve)
    else:
        obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.none_model_vars()

    return obj_fun, time_sol, gap, x_searchers, b_target, threads


# main wrappers
def central_wrapper(g, horizon, searchers, b0, M_target, gamma, timeout):
    """Add variables, constraints, objective function and solve the model
    compute all paths"""

    solver_type = 'central'

    # V^{s, t}
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

                # check for stoppers: either the objective function converged or iterated as much as I wanted
                if (delta_obj < 1e-4) or (my_counter >= n_it):
                    time_sol_list['total'] = total_time_sol
                    # clean and delete

                    disposeDefaultEnv()

                    return obj_fun_list, time_sol_list, gap_list, x_searchers, b_target, threads


def init_wrapper(specs, sim=False):
    """Initialize necessary classes depending on sim or plan only
    default: plan only"""

    solver_data = cp.create_solver_data(specs)
    searchers = cp.create_searchers(specs)
    belief = cp.create_belief(specs)
    target = cp.create_target(specs)

    if sim:
        print('Start target: %d, searcher: %d ' % (target.current_pos, searchers[1].start))
    else:
        print('Start searcher: %d ' % searchers[1].start)

    return belief, searchers, solver_data, target


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

    Tau = ext.get_set_time_u_0(horizon)
    start = ext.get_searchers_positions(searchers)
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


def path_as_list(path: dict):
    """Get sequence of vertices from path[s, t] = v
    for searcher s
    return as list [v0, v1, v2...vh]"""

    pi = dict()

    h = ext.get_h_from_tuple(path)
    m = ext.get_m_from_tuple(path)
    T = ext.get_set_time_u_0(h)
    S = ext.get_set_searchers(m)[0]

    # loop through time
    for s in S:
        pi[s] = []
        for t in T:
            v = path[(s, t)]
            pi[s].append(v)

    return pi


def print_path(x_s: dict):
    pi_dict = xs_to_path(x_s)
    path = path_as_list(pi_dict)

    print('--\nPlanned path: ')
    for s in path.keys():
        path_s = path[s]
        print("Searcher %d: %s" % (s, path_s))

    return path


def path_of_s(path: dict, s_id):
    """Get sequence of vertices [list] for searcher s_id
    INPUT path[s, t] = v
    OUTPUT [v0, ....v]"""

    pi = path_as_list(path)

    return pi[s_id]


def next_from_path(path: dict, t_plan: int):
    """ get new position of searchers as new_pos = {s: v}"""

    m = ext.get_m_from_tuple(path)
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


def searchers_evolve(searchers, new_pos):
    """call to evolve searchers position """

    for s_id in searchers.keys():
        searchers[s_id].evolve_position(new_pos[s_id])

    return searchers
# ----------------------------------------------------------------------------


if __name__ == "__main__":

    planned_path = run_planner()
