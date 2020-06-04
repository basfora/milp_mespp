from core import extract_info as ext
from core import construct_model as cm
from core import analyze_results as ar
from gurobipy import *


def run_gurobi(graph_file, name_folder: str, horizon: int, searchers_info: dict, b0: list, M_target, gamma=0.99):
    """Start and run model based on graph_file (map), deadline (number of time steps)"""

    g = ext.get_graph(graph_file)

    # create model
    md = Model("my_model")

    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers_info)

    # add variables
    my_vars = add_variables(md, g, horizon, start, vertices_t, )

    # add constraints (central algorithm)
    add_constraints(md, g, my_vars, searchers_info, vertices_t, horizon, b0, M_target)

    set_solver_parameters(md, gamma, horizon, my_vars)

    md.update()
    # Optimize model
    md.optimize()

    if GRB.Status == 3:
        print('Optimization problem was infeasible.')
        return False
    elif md.Status == 2:
        # Optimal solution found.
        return True, md
    else:
        print('Unknown error')
        print(md.GRB.Status)
        return False


def add_constraints(md, g, my_vars: dict, searchers_info: dict, vertices_t: dict, deadline: int, b0: list, M: list):
    """define the model constraints according to given parameters
    searchers constraints: (1) - (4)
    :param vertices_t:
    :param searchers_info
    :param my_vars
    :param deadline
    :param b0
    :param M
    :param md
    :param g
    """

    start = ext.get_start_set(searchers_info)

    # searchers motion
    cm.add_searcher_constraints(md, g, my_vars, start, vertices_t, deadline)

    # target motion and intercept events
    cm.add_capture_constraints(md, g, my_vars, searchers_info, vertices_t, b0, M, deadline)


def set_solver_parameters(m, gamma, horizon, my_vars, timeout=30 * 60, pre_solve=-1):
    """ Define my objective function to be maximized """
    h = ext.get_set_time(horizon)
    beta = cm.get_var(my_vars, 'beta')
    m.setObjective(quicksum((gamma ** t) * beta[0, t] for t in h), GRB.MAXIMIZE)

    m.setParam('TimeLimit', timeout)
    m.setParam('Threads', 8)
    m.setParam('Presolve', pre_solve)


def add_variables(md, g, deadline: int, start, vertices_t, s_info=None):

    """Create the variables for my optimization problem
    :param s_info:
    :param start:
    :param vertices_t:
    :param deadline
    """
    # variables related to searcher position and movement
    searchers_vars = cm.add_searcher_variables(md, g, start, vertices_t, deadline)[0]

    # variables related to target position belief and capture
    target_vars = cm.add_target_variables(md, g, deadline, s_info)[0]

    # get my variables together in one dictionary
    my_vars = {}
    my_vars.update(searchers_vars)
    my_vars.update(target_vars)

    return my_vars


def solve_model(md, searchers_info: dict):

    obj_fun, time_sol, gap, s_pos, b_target, threads = none_model_vars()
    threads = 0

    md.update()

    # Optimize model
    md.optimize()

    if md.Status == 3:
        # infeasible solution
        print('Optimization problem was infeasible.')
    elif md.Status == 2:
        # Optimal solution found.
        s_pos, b_target = ar.query_variables(md, searchers_info)
        obj_fun, time_sol, gap, threads = get_model_data(md)
    elif md.Status == 9:
        # time limit reached
        print('Time limit reached.')
        if md.SolCount > 0:
            # retrieve the best solution so far
            s_pos, b_target = ar.query_variables(md, searchers_info)
            obj_fun, time_sol, gap, threads = get_model_data(md)
    else:
        print('Error: ' + str(md.Status))

    # clean things
    # md.reset()
    # md.terminate()
    # disposeDefaultEnv()
    # del md

    return obj_fun, time_sol, gap, s_pos, b_target, threads


def none_model_vars():
    """Return empty model variables
    To keep python from complaining they might been referenced before assignment"""
    obj_fun = None
    time_sol = None
    gap = None
    s_pos = None
    b_target = None
    threads = None

    return obj_fun, time_sol, gap, s_pos, b_target, threads


def get_model_data(md):
    obj_fun = md.objVal
    gap = md.MIPGap
    time_sol = round(md.Runtime, 4)
    threads = md.Params.Threads

    return obj_fun, time_sol, gap, threads


def del_model(model):

    del model
    disposeDefaultEnv()

