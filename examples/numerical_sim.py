# ---------------------------------------------------------------------------------------------------------------------
# start of header
# add module to python path
import sys
import os

import core.plot_fun

this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_path)

# import relevant modules
from core import sim_fun as sf
from classes.inputs import MyInputs
# end of header
# ---------------------------------------------------------------------------------------------------------------------

# initialize default inputs
exp_inputs = MyInputs()
# graph number -- MUSEUM (1), GRID 10x10 (2), 3
exp_inputs.set_graph(2)
# solver parameter: central x distributed
exp_inputs.set_solver_type('distributed')
# searchers' detection: capture range and false negatives
exp_inputs.set_capture_range(1)
exp_inputs.set_zeta(0.3)
# time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
exp_inputs.set_all_times(10)
exp_inputs.set_theta(1)
exp_inputs.set_timeout(5)

# repetitions for each configuration
exp_inputs.set_number_of_runs(100)
# set random seeds
exp_inputs.set_start_seeds(2000, 6000)

# loop for iterative variable m or h
for m in exp_inputs.list_m:
    # update inputs
    exp_inputs.set_size_team(m)

    # loop for number of repetitions
    for turn in exp_inputs.list_turns:

        # create folder to store data
        name_folder = exp_inputs.create_folder()
        # set seed according to run #
        exp_inputs.set_seeds(turn)

        # run simulator
        belief, target, searchers, sim_data = sf.run_simulator(exp_inputs)

        # save everything as a pickle file
        sf.create_save_pickle(belief, target, searchers, sim_data, name_folder, exp_inputs)

        # iterate run #
        today_run = exp_inputs.update_run_number()
        # if wanting to plot
        if turn < 1:
            core.plot_fun.plot_sim_results(belief, target, searchers, sim_data, name_folder)

        # delete things
        del belief, target, searchers, sim_data, name_folder
        print("----------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------")

# clear initial inputs after experiment
del exp_inputs
