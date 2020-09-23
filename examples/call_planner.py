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
# searchers' detection: capture range and false negatives
m = 3
specs.set_capture_range(0)
specs.set_size_team(m)

# time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
specs.set_all_times(5)
specs.set_theta(1)
specs.set_timeout(10)

my_path = pln.run_planner(specs)

