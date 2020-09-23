"""Example for changing specs and calling the planner """

# ---------------------------------------------------------------------------------------------------------------------
# start of header
# import relevant modules
from core import plan_fun as pln
from core import plot_fun as pf
from classes.inputs import MyInputs

# end of header
# ---------------------------------------------------------------------------------------------------------------------

# initialize default inputs
specs = MyInputs()

# load graph, either by number (int), iGraph object or .p file name (str)
specs.set_graph(8)
# solver parameter: central x distributed
specs.set_solver_type('distributed')
# target motion
specs.set_target_motion('static')
# searchers' detection: capture range and false negatives
m = 3
specs.set_capture_range(0)
specs.set_size_team(m)

# time-step stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
h = 5
specs.set_all_times(h)
specs.set_theta(1)
# solver timeout (in sec)
specs.set_timeout(10)

output_solver_data = True
my_path, solver_data = pln.run_planner(specs, output_solver_data)

# to retrieve belief computed by the solver
# we want b(t+1) - next time step
t_next = 1
my_belief_vec = solver_data.retrieve_solver_belief(0, t_next)

