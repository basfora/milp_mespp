# ---------------------------------------------------------------------------------------------------------------------
# start of header
# import relevant modules
from core import plan_fun as pln
from core import plot_fun as pf
from classes.inputs import MyInputs
from core import sim_fun as sf

# end of header
# ---------------------------------------------------------------------------------------------------------------------


def call_planner_ex():
    """Example for changing specs and calling the planner """

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
    h = 10
    specs.set_all_times(h)
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(10)

    # belief
    n = len(specs.graph.vs)
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_0 = [0.0 for i in range(n + 1)]
    for v in v_maybe:
        b_0[v] = 1 / 6
    specs.set_b0(b_0)
    # searchers initial vertices
    specs.set_start_searchers([3, 3, 3])

    output_solver_data = True
    my_path, solver_data = pln.run_planner(specs, output_solver_data)

    # to retrieve belief computed by the solver
    # we want b(t+1) - next time step
    t_next = 1
    my_belief_vec = solver_data.retrieve_solver_belief(0, t_next)

    return my_path, my_belief_vec


def call_sim_ex():
    """Example for changing specs and calling the planner """

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
    h = 10
    specs.set_all_times(h)
    specs.set_deadline(50)
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(120)

    # belief
    n = len(specs.graph.vs)
    v_maybe = [8, 10, 12, 14, 15, 17]
    b_0 = [0.0 for i in range(n + 1)]
    for v in v_maybe:
        b_0[v] = 1 / 6
    specs.set_b0(b_0)
    specs.set_start_target_vertex(5)
    # searchers initial vertices
    specs.set_start_searchers([1, 1, 1])

    # run simulator
    belief, target, searchers, sim_data = sf.run_simulator(specs)




call_sim_ex()
