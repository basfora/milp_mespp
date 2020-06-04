from core import sim_fun as sf
from core import analyze_results as ar

# run simulator with default settings
# sim_param=1, inputs_opt=1, today_run=0, gamma=1.5
# sf.run_simulator()

# INPUTS
# defines simulation parameters: theta, deadline (N), solver_type (central or distributed)
sim_param = 4

# defines: graph to be used, horizon (H), target motion, belief distribution, target and searchers initial vertices
inputs_opt = 4

# today's run
today_run = 2

# cost function parameter
gamma = 1.5

# save data (False for not saving it)
save_data = True
name_folder = None

if save_data is True:
    # name and create folder
    name_folder = ar.create_my_folder(today_run)

# run simulator
belief, target, searchers, sim_data = sf.run_simulator_pmi(sim_param, inputs_opt, gamma, name_folder, save_data)

# print path taken by searchers
print(searchers[1].path_taken)
print(searchers[2].path_taken)

ar.plot_and_show_sim_results(belief, target, searchers, sim_data, name_folder)
