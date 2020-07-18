from core import extract_info as ext
from core import construct_model as cm
from core import create_parameters as cp
from core import plan_fun as pln

import pickle
import random
import os


def run_simulator(specs=None):
    """Initialize the planner the pre-set parameters
    If needed, change parameters here using MyInputs() class functions
    Return path of searchers as list"""

    if specs is None:
        specs = cp.default_specs()

    belief, target, searchers, solver_data = simulator_main(specs)

    return belief, target, searchers, solver_data


def simulator_main(specs, printout=True):

    # extract inputs for the problem instance
    timeout = specs.timeout
    g = specs.graph
    m = specs.size_team

    # initialize classes
    belief, searchers, solver_data, target = pln.init_wrapper(specs, True)
    # -------------------------------------------------------------------------------

    deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
    M = target.unpack()

    # get sets for easy iteration
    S, V, _, m, n = ext.get_sets_and_ranges(g, m, horizon)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    path = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        print('--\nTime step %d \n--' % t)

        # _________________
        if t % theta == 0:
            # check if it's time to re-plan (OBS: it will plan on t = 0)

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_s, b_target, threads = pln.run_solver(g, horizon, searchers, belief.new,
                                                                                    M, gamma, solver_type, timeout)

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_s, b_target, t)

            # break here if the problem was infeasible
            if time_sol is None or gap is None or obj_fun is None:
                break

            # get position of each searcher at each time-step based on x[s, v, t] variable to path [s, t] = v
            searchers, path = pln.update_plan(searchers, x_s)

            if printout:
                pln.print_path(x_s)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        if printout:
            # print current positions
            print('t = %d' % t)
            print_positions(searchers, target)

        path_next_t = pln.next_from_path(path, t_plan)

        # evolve searcher position
        searchers = pln.searchers_evolve(searchers, path_next_t)

        # update belief
        belief.update(searchers, path_next_t, M, n)

        # update target
        target = evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = check_for_capture(searchers, target)

        if (t == deadline) and printout:
            print('--\nTime step %d\n--' % deadline)
            print('t = %d' % t)
            print_positions(searchers, target)

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
def print_positions(searchers, target):
    """Print current searchers and target's position"""

    print('Target vertex: %d' % target.current_pos)

    for s_id in searchers.keys():
        s = searchers[s_id]
        print('Searcher %d: vertex %d' % (s_id, s.current_pos), sep=' ', end=' ', flush=True)
    print('\n')

    return


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
    if not os.path.exists(file_path):
        os.mkdir(file_path)
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


def evolve_target(target, updated_belief: list):
    """evolve possible and true position of the target"""
    # evolve target true position (based on motion matrix)
    target.evolve_true_position()
    # evolve target possible positions for next time, according to new belief
    target.evolve_possible_position(updated_belief)

    return target


if __name__ == "__main__":
    belief_obj, target_obj, searchers_obj, solver_data_obj = run_simulator()

