# start of header
# add module to python path
import sys
import os
import numpy as np
this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_path)
from gurobipy import *
from core import extract_info as ext
from core import sim_fun as sf
from core import retrieve_data as rd
from core.classes.class_inputs import MyInputs
from core.classes.class_belief import MyBelief
from core.classes.class_searcher import MySearcher
from core.classes.class_target import MyTarget
from core import create_parameters as cp


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
    disposeDefaultEnv()

    t = 0
    # get sets for easy iteration
    S, V, Tau, n, m = ext.get_sets_and_ranges(specs.graph, specs.size_team, specs.horizon)

    belief, target, searchers, solver_data, s_info = sf.my_init_wrapper(specs)

    M = sf.unpack_from_target(target)
    timeout = specs.timeout

    # ------------------------------------------
    # call for model solver wrapper according to centralized or decentralized solver and return the solver data
    obj_fun, time_sol, gap, x_searchers, b_target, threads = sf.run_solver(specs.graph, specs.horizon, s_info,
                                                                           belief.new, M, specs.gamma,
                                                                           specs.solver_type, timeout)

    # save the new data
    solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_searchers, b_target, t)

    # break here if the problem was infeasible
    if time_sol is None or gap is None or obj_fun is None:
        # belief, target, searchers, solver_data
        return None

    # get position of each searcher at each time-step based on x[s][v, t] variable
    searchers, pi_s = sf.get_planned_path(x_searchers, V, Tau, searchers)

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











