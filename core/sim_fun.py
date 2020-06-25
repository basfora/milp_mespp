from core import create_parameters as cp
from core import extract_info as ext
from core import construct_model as cm
from core import plan_fun as pln

from classes.belief import MyBelief
from classes.target import MyTarget
from classes.searcher import MySearcher
from classes.solver_data import MySolverData

import pickle
import random
from gurobipy import *
from core.deprecated import pre_made_inputs as pmi


def run_my_simulator(exp_inputs):

    # INITIALIZE

    # make sure Gurobi env is clean
    disposeDefaultEnv()
    # extract inputs for the problem instance
    timeout = exp_inputs.timeout
    g = exp_inputs.graph
    m = exp_inputs.size_team

    # initialize classes
    belief, target, searchers, solver_data = my_init_wrapper(exp_inputs)
    # -------------------------------------------------------------------------------

    deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
    M = target.unpack()

    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(g, m, horizon)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    pi_s = {}
    pi_next_t = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        print('--\nTime step %d \n--' % t)

        # _________________
        # check if it's time to re-plan (OBS: it will plan on t = 0)
        if t % theta == 0:

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, searchers, belief.new, M,
                                                                                     gamma, solver_type, timeout)

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

            # break here if the problem was infeasible
            if time_sol is None or gap is None or obj_fun is None:
                break

            # get position of each searcher at each time-step based on x[s][v, t] variable
            searchers, pi_s = pln.xs_to_path(x_searchers, V, Tau, searchers)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        pi_next_t = get_new_pos(searchers, pi_s, t_plan)

        # evolve searcher position
        searchers = evolve_searchers_position(searchers, pi_next_t)

        # update belief
        belief.update(searchers, pi_next_t, M, n)

        # update target
        target = evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = check_for_capture(searchers, target)

        if target.is_captured:
            print_capture_details(t, target, searchers, solver_data)
            break

    return belief, target, searchers, solver_data

#
#
#
#
# ---------------------------------------------------------------------------------------------------------------------
# END OF SIMULATION FLOW
#
# ---------------------------------------------------------------------------------------------------------------------
#
#
# auxiliary functions


def my_init_wrapper(exp_inputs):

    g = exp_inputs.graph

    # planning stuff
    deadline = exp_inputs.deadline
    theta = exp_inputs.theta
    horizon = exp_inputs.horizon
    solver_type = exp_inputs.solver_type

    capture_range = exp_inputs.capture_range
    m = exp_inputs.size_team
    zeta = exp_inputs.zeta

    s_seed = exp_inputs.searcher_seed

    # target stuff
    target_motion = exp_inputs.target_motion
    t_possible_nodes = exp_inputs.qty_possible_nodes
    t_seed = exp_inputs.target_seed

    # gather seeds
    my_seed = dict()
    my_seed['searcher'] = s_seed
    my_seed['target'] = t_seed

    # belief stuff
    belief_distribution = exp_inputs.belief_distribution

    init_is_ok = False
    v_target, v_searchers = None, None

    while init_is_ok is False:

        v_searchers, v_target = cp.random_init_pos(g, m, t_possible_nodes, my_seed)

        init_is_ok = cp.check_reachability(g, capture_range, v_target, v_searchers)

        if init_is_ok is False:
            print('Target within range --> target: ' + str(v_target) + 'searcher ' + str(v_searchers))
            my_seed['searcher'] = my_seed['searcher'] + 500

    # initialize parameters if everything is ok
    b0, M, s_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution, capture_range,
                                       zeta)

    searchers = cp.create_searchers(g, v_searchers, capture_range, zeta)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, solver_data = init_all_classes(horizon, deadline, theta, g, solver_type, b0, s_info,
                                                   v_target, M, capture_range, my_seed)

    print('Start target: %d, searcher: %d ' % (target.current_pos, searchers[1].start))

    return belief, target, searchers, solver_data


def print_capture_details(t, target, searchers, solver_data):
    """Print capture details on terminal"""
    print("\nCapture details: \ntime = " + str(t), "\nvertex = " + str(target.capture_v),
          "\nsearcher = " + str(ext.find_captor(searchers)))

    print("Solving time: ", solver_data.solve_time)
    return


def create_save_pickle(belief, target, searchers, solver_data, name_folder: str, exp_inputs=None, parent_folder='data'):
    """Create dictionary with belief, target, searchers and solver data
    dump in pickle file"""

    # name the pickle file
    file_path = ext.get_whole_path(name_folder, parent_folder)
    file_name = 'global_save.txt'
    full_path = file_path + "/" + file_name

    exp_data = dict()
    exp_data["belief"] = belief
    exp_data["target"] = target
    exp_data["searchers"] = searchers
    exp_data["solver_data"] = solver_data
    if exp_inputs is not None:
        exp_data["inputs"] = exp_inputs

    my_pickle = open(full_path, "wb")
    pickle.dump(exp_data, my_pickle)
    my_pickle.close()

    print("Data saved in: ", name_folder)
    return


def load_pickle_file(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b


def check_for_capture(searchers, target):
    """Check if the next position of any searcher
    allow it to capture the target"""

    # get next time and vertex (after evolving position)
    next_time, v_target = ext.get_last_info(target.stored_v_true)

    for s_id in searchers.keys():
        s = searchers[s_id]
        zeta = s.zeta
        # get capture matrix for that vertex
        C = s.capture_matrices.get(s.current_pos)

        # just check for capture
        if cm.check_capture(C, v_target):
            # no false negatives
            if zeta is None:
                zeta = 0
            chance_capture = random.random()
            # flip a coin
            if chance_capture <= 1 - zeta:

                target.update_status(True)
                searchers[s_id].update_status(True)
                break

    return searchers, target


def init_all_classes(horizon: int, deadline: int, theta: int, g, solver_type: str, init_belief: list,
                     searchers_info: dict, v_possible_target: list, motion_matrix_target: list,
                     capture_range=0, my_seeds=None, idx_true_target=None):
    """Initialize all classes:
    belief(init_belief)
    searcher(searchers_info)
    target(v_possible, motion_matrix, vtrue)
    solverdata(horizon)"""

    # initialize classes
    sim_data = MySolverData(horizon, deadline, theta, g, solver_type)

    # belief
    belief = MyBelief(init_belief)

    if my_seeds is not None:
        my_t_seed = my_seeds['target']
        my_s_seed = my_seeds['searcher']
    else:
        my_s_seed, my_t_seed = None, None

    # target
    v_target_true = cm.get_target_true_position(v_possible_target, idx_true_target)
    target = MyTarget(v_possible_target, motion_matrix_target, v_target_true, my_t_seed)

    m = len(searchers_info.keys())

    return belief, target, sim_data


def init_searchers(searchers_info, capture_range, m, my_s_seed=None):
    searchers = {}
    for s_id in searchers_info.keys():
        searcher = MySearcher(s_id, searchers_info, capture_range, m, my_s_seed)
        searchers[s_id] = searcher
    return searchers


def update_start_info(searcher_info: dict, current_position: dict):
    """update the start position of the searchers for the planning algorithm"""

    for s_id in searcher_info.keys():
        searcher_info[s_id]['start'] = current_position[s_id]

    return searcher_info


def evolve_searchers_position(searchers, new_pos):
    """call to evolve searchers position """

    for s_id in searchers.keys():
        searchers[s_id].evolve_position(new_pos[s_id])

    return searchers


def get_new_pos(searchers, s_pos_plan: dict, t_plan: int):
    """ get new position of searchers as new_pos = {s: v}"""
    new_pos = {}
    for s_id in searchers.keys():
        new_pos[s_id] = s_pos_plan[s_id, t_plan]

    return new_pos


def evolve_target(target, updated_belief: list):
    """evolve possible and true position of the target"""
    # evolve target true position (based on motion matrix)
    target.evolve_true_position()
    # evolve target possible positions for next time, according to new belief
    target.evolve_possible_position(updated_belief)

    return target


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# OLD VERSIONS
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
def run_simulator_pmi(sim_param=1, inputs_opt=1, gamma=0.99, name_folder=None, save_data=True):
    """Run the simulator given option of pre-made:
    simulation parameters (replan frequency, deadline, name of folder to save data)
    inputs (graph, initial vertex for target and searchers
    number of runs for today (for naming the folder where data will be saved)
    gamma value for the objective function
    option for the solver: central or distributed """

    # _______________________________________________________________________________________________________________

    # GET parameters for the simulation, according to sim_param
    horizon, theta, deadline, solver_type = pmi.get_sim_param(sim_param)

    # GET parameters for the MILP solver
    # get horizon, graph etc according to inputs_opt parameter (pre-made inputs)
    g, v0_target, v0_searchers, target_motion, belief_distribution = pmi.get_inputs(inputs_opt)

    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(v0_searchers, deadline, g)

    capture_range = 0

    # ________________________________________________________________________________________________________________

    # INITIALIZE

    # initialize parameters according to inputs
    b_0, M, s_info = cp.init_parameters(g, v0_target, v0_searchers, target_motion, belief_distribution)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, solver_data = init_all_classes(horizon, deadline, theta, g, solver_type, b_0,  s_info,
                                                   v0_target, M, capture_range)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    pi_s = {}
    pi_next_t = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        # _________________
        # check if it's time to re-plan (OBS: it will plan on t = 0)
        if t % theta == 0:

            # update start vertices of the searchers
            if t is not 0:
                s_info = update_start_info(s_info, pi_next_t)

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, s_info, belief.new, M, gamma,
                                                                                solver_type)

            # break here if the problem was infeasible
            if time_sol is None:
                return None, None, None, None

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

            # get position of each searcher at each time-step based on x[s][v, t] variable
            searchers, pi_s = pln.get_planned_path(x_searchers, V, Tau, searchers)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        pi_next_t = get_new_pos(searchers, pi_s, t_plan)

        # evolve searcher position
        searchers = evolve_searchers_position(searchers, pi_next_t)

        # update belief
        belief.update(s_info, pi_next_t, M, n)

        # update target
        target = evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = check_for_capture(searchers, target)

        if target.is_captured:
            print_capture_details(t, target, searchers, solver_data)
            break

    if save_data:
        # save everything as a pickle file
        create_save_pickle(belief, target, searchers, solver_data, name_folder)

    return belief, target, searchers, solver_data


def run_simulator_root(g, m, deadline, theta=None, horizon=None, solver_type='central', target_motion='random',
                       belief_distribution='uniform', target_possible_nodes=5, my_seed=None, capture_range=0,
                       zeta=None, gamma=0.99, name_folder=None, save_data=True):
    """Run the simulator given option of pre-made:
    simulation parameters (replan frequency, deadline, name of folder to save data)
    inputs (graph, initial vertex for target and searchers
    number of runs for today (for naming the folder where data will be saved)
    gamma value for the objective function
    option for the solver: central or distributed """

    # _______________________________________________________________________________________________________________

    if theta is None:
        theta = deadline
        horizon = deadline
    # ________________________________________________________________________________________________________________

    # INITIALIZE

    belief, target, searchers, solver_data, s_info = init_wrapper(g, deadline, horizon, theta, solver_type, m,
                                                                  target_possible_nodes, my_seed, target_motion,
                                                                  belief_distribution, capture_range, zeta)
    # unpack
    M = target.motion_matrix

    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(g, m, horizon)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    pi_s = {}
    pi_next_t = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        # _________________
        # check if it's time to re-plan (OBS: it will plan on t = 0)
        if t % theta == 0:

            # update start vertices of the searchers
            if t is not 0:
                s_info = update_start_info(s_info, pi_next_t)

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, s_info, belief.new, M,
                                                                                gamma, solver_type)

            # break here if the problem was infeasible
            if time_sol is None:
                return None, None, None, None

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

            # get position of each searcher at each time-step based on x[s][v, t] variable
            searchers, pi_s = pln.get_planned_path(x_searchers, V, Tau, searchers)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        pi_next_t = get_new_pos(searchers, pi_s, t_plan)

        # evolve searcher position
        searchers = evolve_searchers_position(searchers, pi_next_t)

        # update belief
        belief.update(s_info, pi_next_t, M, n)

        # update target
        target = evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = check_for_capture(searchers, target)

        if target.is_captured:
            print_capture_details(t, target, searchers, solver_data)
            break

    if save_data:
        # save everything as a pickle file
        create_save_pickle(belief, target, searchers, solver_data, name_folder)

    return belief, target, searchers, solver_data


def run_simulator_module(g, plan_input: dict, searcher_input: dict, target_input: dict, belief_input: dict):
    """Run the simulator given option of pre-made:
    simulation parameters (replan frequency, deadline, name of folder to save data)
    inputs (graph, initial vertex for target and searchers
    number of runs for today (for naming the folder where data will be saved)
    gamma value for the objective function
    option for the solver: central or distributed """

    # INITIALIZE

    belief, target, searchers, solver_data, s_info = init_wrapper_mod(g, plan_input, searcher_input,
                                                                      target_input, belief_input)

    deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
    M = target.unpack()

    m = searcher_input['size_team']

    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(g, m, horizon)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    pi_s = {}
    pi_next_t = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        # _________________
        # check if it's time to re-plan (OBS: it will plan on t = 0)
        if t % theta == 0:

            # update start vertices of the searchers
            if t is not 0:
                s_info = update_start_info(s_info, pi_next_t)

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_searchers, b_target, threads = pln.run_solver(g, horizon, s_info, belief.new, M,
                                                                                gamma, solver_type)

            # break here if the problem was infeasible
            if time_sol is None:
                return None, None, None, None

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

            # get position of each searcher at each time-step based on x[s][v, t] variable
            searchers, pi_s = pln.get_planned_path(x_searchers, V, Tau, searchers)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        pi_next_t = get_new_pos(searchers, pi_s, t_plan)

        # evolve searcher position
        searchers = evolve_searchers_position(searchers, pi_next_t)

        # update belief
        belief.update(s_info, pi_next_t, M, n)

        # update target
        target = evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = check_for_capture(searchers, target)

        if target.is_captured:
            print_capture_details(t, target, searchers, solver_data)
            break

    return belief, target, searchers, solver_data


def init_wrapper(g, deadline, horizon, theta, solver_type, m: int, t_possible_nodes: int, my_seed,
                 target_motion='random', belief_distribution='uniform', capture_range=0, zeta=None):
    """Initialize the parameters and classes
    Return the classes and the searchers_info
    :param g -- graph
    :param horizon -- horizon of planning
    :param deadline -- deadline for simulation
    :param theta -- replan frequency
    :param m -- number of searchers
    :param t_possible_nodes -- number of possible nodes the target might be in
    :param my_seed -- seed to retrieve random position
    :param target_motion -- type of target motion, default is random, accepts also static
    :param belief_distribution -- type of belief distribution, accepts 'uniform'
    :param capture_range -- searchers capture range, default is zero (same vertex)
    :param zeta: false negative in [0, 1], default if None (no false negatives)
    :param """

    init_is_ok = False
    v_target, v_searchers, my_seeds = None, None, None

    while init_is_ok is False:
        v_target, v_searchers, my_seeds = cp.random_init_pos(g, m, t_possible_nodes, my_seed)

        init_is_ok = cp.check_reachability(g, capture_range, v_target, v_searchers)

        my_seed += 200

    # initialize parameters if everything is ok
    b0, M, s_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution, capture_range, zeta)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, searchers, solver_data = init_all_classes(horizon, deadline, theta, g, solver_type, b0, s_info,
                                                              v_target, M, capture_range, my_seeds)

    return belief, target, searchers, solver_data, s_info



def init_wrapper_mod(g, plan_input: dict, searcher_input: dict, target_input: dict, belief_input: dict):
    """Initialize the parameters and classes
    Return the classes and the searchers_info
    :param g -- graph
    :param plan_input
    :param searcher_input
    :param target_input
    :param belief_input
    """

    # planning stuff
    deadline = plan_input['deadline']
    theta = plan_input['theta']
    horizon = plan_input['horizon']
    solver_type = plan_input['solver_type']

    if theta is None:
        theta = deadline
        horizon = deadline

    # searcher stuff
    capture_range = searcher_input['capture_range']
    m = searcher_input['size_team']
    zeta = searcher_input['zeta']
    s_seed = searcher_input['seed']

    # target stuff
    target_motion = target_input['target_motion']
    t_possible_nodes = target_input['qty_possible_nodes']
    t_seed = target_input['seed']

    # gather seeds
    my_seed = dict()
    my_seed['searcher'] = s_seed
    my_seed['target'] = t_seed

    # belief stuff
    belief_distribution = belief_input['belief_distribution']

    init_is_ok = False
    v_target, v_searchers = None, None

    while init_is_ok is False:
        v_searchers, v_target = cp.random_init_pos(g, m, t_possible_nodes, my_seed)

        init_is_ok = cp.check_reachability(g, capture_range, v_target, v_searchers)

        my_seed['searcher'] += 21

    # initialize parameters ikf everything is ok
    b0, M, s_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution, capture_range, zeta)

    # initialize instances of classes (initial target and searchers locations)
    belief, target, searchers, solver_data = init_all_classes(horizon, deadline, theta, g, solver_type, b0, s_info,
                                                              v_target, M, capture_range, my_seed)

    return belief, target, searchers, solver_data, s_info

