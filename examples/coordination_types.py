"""Visual example between explicit and implicit coordination algorithms"""
from milp_mespp.core import create_parameters as cp
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import plot_fun as pf


def choose_parameters():
    """Set simulation parameters"""
    specs = MyInputs()
    # searchers: default capture range = 0 (single vertex), zeta = 0 (no false negatives)
    # team size
    m = 3
    # planning horizon
    h = 2
    # graph: 4 x 4 grid
    graph = cp.create_grid_graph(4, 4)
    n = len(graph.vs)
    # target: default
    t0 = [5, 8, 10, 12]

    # set it up
    specs.set_size_team(m)
    specs.set_all_times(h)
    specs.set_graph(graph)
    specs.set_start_target(t0)

    return specs


def plan_example(specs, solver_type='central'):
    """Implicit coordination (distributed) algorithm"""

    specs.set_solver_type(solver_type)
    name_folder = specs.create_folder()

    output_data = True
    path_list, solver_data = pln.run_planner(specs, output_data)

    pf.plot_plan_results(solver_data, name_folder, specs.start_target_true)

    return path_list


def my_script():
    # set parameters
    specs = choose_parameters()

    # get path for explicit coordination
    plan_example(specs)
    print('==========\nImplicit coordination\n==========')
    # get path for implicit coordination
    plan_example(specs, 'distributed')


if __name__ == "__main__":
    my_script()

