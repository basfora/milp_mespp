"""Solver-related functions
 - add variables
 - add constraints
 - set objective function
 - solve model
 - retrieve model data
 - query variables

OBS: s in {1,...m}
G(V, E), v in {1,...n} but igraph treats as zero
t in tau = {1,...T}
capture matrix: first index is zero [0][0], first vertex is [1][1]
v0 can be {1,...n} --> I convert to python indexing later
array order: [s, t, v, u]"""

from core import extract_info as ext
from core import construct_model as cm
from gurobipy import *


def add_constraints(md, g, my_vars: dict, searchers: dict, vertices_t: dict, deadline: int, b0: list, M: list):
    """define the model constraints according to given parameters
    searchers constraints: (1) - (4)
    :param vertices_t:
    :param searchers
    :param my_vars
    :param deadline
    :param b0
    :param M
    :param md
    :param g
    """

    start = ext.get_start_set(searchers)

    # searchers motion
    add_searcher_constraints(md, g, my_vars, start, vertices_t, deadline)

    # target motion and intercept events
    add_capture_constraints(md, g, my_vars, searchers, vertices_t, b0, M, deadline)


def set_solver_parameters(m, gamma, horizon, my_vars, timeout=30 * 60, pre_solve=-1):
    """ Define my objective function to be maximized """
    h = ext.get_set_time(horizon)
    beta = get_var(my_vars, 'beta')
    m.setObjective(quicksum((gamma ** t) * beta[0, t] for t in h), GRB.MAXIMIZE)

    m.setParam('TimeLimit', timeout)
    m.setParam('Threads', 8)
    m.setParam('Presolve', pre_solve)


def add_variables(md, g, deadline: int, start, vertices_t, searchers=None):
    # TODO IMPORTANT change to allow for different zetas!!

    """Create the variables for my optimization problem
    :param searchers:
    :param start:
    :param vertices_t:
    :param deadline
    """
    # variables related to searcher position and movement
    searchers_vars = add_searcher_variables(md, g, start, vertices_t, deadline)[0]

    # variables related to target position belief and capture
    target_vars = add_target_variables(md, g, deadline, searchers)[0]

    # get my variables together in one dictionary
    my_vars = {}
    my_vars.update(searchers_vars)
    my_vars.update(target_vars)

    return my_vars


def solve_model(md):

    obj_fun, time_sol, gap, x_searchers, b_target, threads = none_model_vars()
    threads = 0

    md.update()

    # Optimize model
    md.optimize()

    if md.Status == 3:
        # infeasible solution
        print('Optimization problem was infeasible.')
    elif md.Status == 2:
        # Optimal solution found.
        x_searchers, b_target = query_variables(md)
        obj_fun, time_sol, gap, threads = get_model_data(md)
    elif md.Status == 9:
        # time limit reached
        print('Time limit reached.')
        if md.SolCount > 0:
            # retrieve the best solution so far
            x_searchers, b_target = query_variables(md)
            obj_fun, time_sol, gap, threads = get_model_data(md)
    else:
        print('Error: ' + str(md.Status))

    # clean things
    # md.reset()
    # md.terminate()
    # disposeDefaultEnv()
    # del md

    return obj_fun, time_sol, gap, x_searchers, b_target, threads


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


def add_searcher_variables(md, g, start: list, vertices_t: dict, deadline: int):
    """Add variables related to the searchers on the model:
    searchers location at each time step, X
    searchers movement from t to t + 1, Y"""

    [X, Y] = ext.init_dict_variables(2)

    S, m = ext.get_set_searchers(start)

    var_for_test = {}
    list_x_name = []
    list_y_name = []

    # Tau_ = ext.get_idx_time(deadline)
    Tau_ext = ext.get_set_ext_time(deadline)

    # searcher s is located at vertex v at time t
    for s in S:
        for t in Tau_ext:
            # get vertices that searcher s can be at time t (last t needs to be dummy goal vertex)
            # v_t returns the label of the vertices
            v_t = vertices_t.get((s, t))
            for v in v_t:
                # variable for: searcher position --  for each node the searcher can be at each time
                dummy_x_name = "x[%d,%d,%d]" % (s, v, t)
                if dummy_x_name not in list_x_name:
                    # if didn't add already, do it
                    X[s, v, t] = md.addVar(vtype="BINARY", name=dummy_x_name)
                    list_x_name.append(dummy_x_name)

                # find Y[s, u, v, t] : from u, searcher s can move to u at t + 1
                # returns vertices labels
                my_next_v = cm.get_next_vertices(g, s, v, t, vertices_t, Tau_ext)
                if my_next_v is not None:
                    for u in my_next_v:
                        dummy_y_name = "y[%d,%d,%d,%d]" % (s, v, u, t)
                        if dummy_y_name not in list_y_name:
                            Y[s, v, u, t] = md.addVar(vtype="BINARY", name=dummy_y_name)
                            list_y_name.append(dummy_y_name)

    var_for_test.update({'x': list_x_name})
    var_for_test.update({'y': list_y_name})

    my_vars = {'x': X, 'y': Y}
    return my_vars, var_for_test


def add_target_variables(md, g, deadline: int, searchers=None):
    """Add variables related to the target and capture events:
    belief variable, B
    interception-related variables: belief vector composition, beta
    belief evolution, alpha
    capture event, zeta and psi
    """
    # TODO test this
    # TODO change this to allow for different zetas

    V = ext.get_set_vertices(g)[0]

    Tau_ext = ext.get_set_ext_time(deadline)
    Tau = ext.get_set_time(deadline)

    V_ext = ext.get_set_ext_vertices(g)

    var_for_test = {}
    list_beta_name = []
    list_alpha_name = []
    list_delta_name = []

    [beta, beta_s, alpha, psi, delta] = ext.init_dict_variables(5)

    if searchers is not None:
        false_neg, zeta = cm.check_false_negatives(searchers)
        S = ext.get_set_searchers(searchers)[0]
    else:
        false_neg = False
        S = None

    # alpha and psi: only exist from 1, 2.., T
    for t in Tau:
        for v in V:
            dummy_a_name = "[%d,%d]" % (v, t)
            alpha[v, t] = md.addVar(vtype="CONTINUOUS", lb=0.0, ub=1.0, name="alpha" + dummy_a_name)

            list_alpha_name.append(dummy_a_name)

            if false_neg:
                for s in S:
                    dummy_delta_name = "[%d,%d,%d]" % (s, v, t)
                    psi[s, v, t] = md.addVar(vtype="BINARY", name="psi" + dummy_delta_name)
                    delta[s, v, t] = md.addVar(vtype="CONTINUOUS", lb=0.0, ub=1.0, name="delta" + dummy_delta_name)

                    list_delta_name.append(dummy_delta_name)
            else:
                psi[v, t] = md.addVar(vtype="BINARY", name="psi" + dummy_a_name)

    for t in Tau_ext:
        for v in V_ext:
            dummy_b_name = "[%d,%d]" % (v, t)
            list_beta_name.append(dummy_b_name)

            beta[v, t] = md.addVar(vtype="CONTINUOUS", lb=0.0, ub=1.0, name="beta" + dummy_b_name)

            if false_neg:
                # include 0 for searcher s = 1, s-1 = 0
                for s_ in [0] + S:
                    dummy_bs_name = "[%d,%d,%d]" % (s_, v, t)
                    beta_s[s_, v, t] = md.addVar(vtype="CONTINUOUS", lb=0.0, ub=1.0, name="beta_s" + dummy_bs_name)

    var_for_test.update({'beta': list_beta_name})
    var_for_test.update({'alpha': list_alpha_name})

    if false_neg:
        my_vars = {'beta': beta, 'beta_s': beta_s, 'alpha': alpha, 'psi': psi, 'delta': delta}
        var_for_test.update({'delta': list_delta_name})
    else:
        my_vars = {'beta': beta, 'alpha': alpha, 'psi': psi}

    return my_vars, var_for_test


def add_searcher_constraints(md, g, my_vars: dict, start: list, vertices_t: dict, deadline: int):
    """Define constraints pertinent to the searchers path """
    # get variables
    X = get_var(my_vars, 'x')
    Y = get_var(my_vars, 'y')

    S, m = ext.get_set_searchers(start)
    Tau_ext = ext.get_set_ext_time(deadline)

    # legality of the paths, for all s = {1,...m}
    for s in S:
        # 0, 1, 2... T
        for t in Tau_ext:
            v_t = vertices_t.get((s, t))
            # each searcher can only be at one place at each time (including the start vertex), Eq. (1, 7)
            if t == 0:
                md.addConstr(X[s, v_t[0], 0] == 1)
            # md.addConstr(quicksum(X[s, v, t] for v in v_t) == 1)

            for u in v_t:
                my_next_v = cm.get_next_vertices(g, s, u, t, vertices_t, Tau_ext)
                my_previous_v = cm.get_previous_vertices(g, s, u, t, vertices_t)
                if my_next_v is not None:
                    # (Eq. 9) searcher can only move to: i in delta_prime(v) AND V^tau(t+1)
                    # sum == 1 if searcher is at u, sum == zero if searcher is not at u (depends on X[s, u, t])
                    md.addConstr(quicksum(Y[s, u, i, t] for i in my_next_v) == X[s, u, t])

                    if my_previous_v is not None:
                        # (Eq. 8) searcher can only move to v from j in delta_prime(v) AND V^tau(t-1)
                        md.addConstr(quicksum(Y[s, i, u, t - 1] for i in my_previous_v) == X[s, u, t])
                        # md.addConstr(quicksum(Y[s, i, u, t - 1] for i in my_previous_v) ==
                                     # quicksum(Y[s, u, i, t] for i in my_next_v))


def add_capture_constraints(md, g, my_vars: dict, searchers: dict, vertices_t, b0: list, M: list, deadline: int):
    """Define constraints about belief and capture events
    :param vertices_t:
    :param md:
    :param g:
    :param my_vars:
    :param searchers:
    :param b0
    :param deadline
    """

    # capture-related variables
    beta = get_var(my_vars, 'beta')
    alpha = get_var(my_vars, 'alpha')
    psi = get_var(my_vars, 'psi')

    false_neg, zeta = cm.check_false_negatives(searchers)

    # if false negative model, there will exist a delta
    if false_neg:
        delta = get_var(my_vars, 'delta')
        beta_s = get_var(my_vars, 'beta_s')
    else:
        delta = {}
        beta_s = {}

    # searchers position
    X = get_var(my_vars, 'x')

    # sets
    V = ext.get_set_vertices(g)[0]
    S, m = ext.get_set_searchers(searchers)

    Tau = ext.get_set_time(deadline)
    V_ext = ext.get_set_ext_vertices(g)

    # initial belief (user input), t = 0 (Eq. 13)
    for i in V_ext:
        md.addConstr(beta[i, 0] == b0[i])

    for t in Tau:
        # this is a dictionary
        my_vertices = cm.get_current_vertices(t, vertices_t, S)
        for v in V:
            # v_idx = ext.get_python_idx(v)
            # take Markovian model into account (Eq. 14)
            # NOTE M matrix is accessed by python indexing
            md.addConstr(alpha[v, t] == quicksum(M[u - 1][v - 1] * beta[u, t - 1] for u in V))

            # find searchers position that could capture the target while it is in v
            list_u_capture = cm.get_u_for_capture(searchers, V, v)
            if list_u_capture and my_vertices:
                if false_neg:
                    for s in S:
                        md.addConstr(quicksum(X[s, u, t] for u in filter(lambda x: x in my_vertices[s], list_u_capture)) >= psi[s, v, t])
                        md.addConstr(quicksum(X[s, u, t] for u in filter(lambda x: x in my_vertices[s], list_u_capture)) <= psi[s, v, t])

                else:
                    md.addConstr(quicksum(quicksum(X[s, u, t] for u in filter(lambda x: x in my_vertices[s], list_u_capture)) for s in S) >= psi[v, t])
                    md.addConstr(quicksum(quicksum(X[s, u, t] for u in filter(lambda x: x in my_vertices[s], list_u_capture)) for s in S) <= m * psi[v, t])

            if false_neg:
                for s in S:

                    # first searcher
                    if s == S[0]:
                        md.addConstr(beta_s[0, v, t] == alpha[v, t])

                    # Eq. (38)
                    md.addConstr(delta[s, v, t] <= 1 - psi[s, v, t])
                    # Eq. (39)
                    md.addConstr(delta[s, v, t] <= beta_s[s-1, v, t])
                    # Eq. (40)
                    md.addConstr(delta[s, v, t] >= beta_s[s-1, v, t] - psi[s, v, t])

                    # Eq. (37)
                    md.addConstr(beta_s[s, v, t] == ((1 - zeta) * delta[s, v, t]) + (zeta * beta_s[s-1, v, t]))

                # last searcher
                md.addConstr(beta[v, t] == beta_s[S[-1], v, t])

            else:
                # (15)
                md.addConstr(beta[v, t] <= 1 - psi[v, t])
                # (16)
                md.addConstr(beta[v, t] <= alpha[v, t])
                # (17)
                md.addConstr(beta[v, t] >= alpha[v, t] - psi[v, t])

        # probability of being intercepted == what is left
        md.addConstr(beta[0, t] == 1 - quicksum(beta[v, t] for v in V))


def get_var(my_vars: dict, name: str):
    """Retrieve variable from my_var dictionary according to name"""
    desired_var = my_vars.get(name)
    if desired_var is not None:
        return desired_var
    else:
        var_names = 'x, y, alpha, beta, zeta, psi'
        print('No variable with this name, current model accepts only:' + var_names)
        return None


def query_variables(md):
    """query variable X to get the optimal path
    query variable beta to get target belief
    return variables values as dictionaries
    x_searchers[(s, v, t)], b_target[(v, t)]"""

    # save as dictionaries with searchers as keys
    x_searchers = {}
    b_target = {}

    t_max = 0

    for var in md.getVars():
        my_var_name = var.varName
        my_var_value = var.x
        # print('%s %g' % (my_var_name, my_var_value))

        if 'x' in my_var_name:
            s = int(my_var_name[2:my_var_name.find(",")])
            v = int(my_var_name[my_var_name.find(",") + 1:my_var_name.rfind(",")])
            t = int(my_var_name[my_var_name.rfind(",") + 1:-1])

            # print('%s = %f ' % (my_var_name, my_var_value))
            x_searchers[(s, v, t)] = my_var_value

            if t > t_max:
                t_max = t

        elif 'beta' in my_var_name and '_s' not in my_var_name:
            # print('%s %g' % (my_var_name, my_var_value))
            # remember b[0] is probability of capture
            v = int(my_var_name[5:my_var_name.find(",")])
            t = int(my_var_name[my_var_name.find(",") + 1:my_var_name.rfind("]")])
            b_target[(v, t)] = my_var_value

    # make sure x is binary
    x_searchers = enforce_binary(x_searchers, t_max)
    b_target = enforce_sum_1(b_target, t_max)

    # x_searchers[(s, v, t)] and b_target[(v, t)]
    return x_searchers, b_target


def enforce_binary(x_searchers, t_max):
    """Enforce variable to be binary"""

    old_x_searchers = x_searchers

    m = ext.get_m_from_xs(x_searchers)
    S = ext.get_set_searchers(m)[0]

    x_keys = old_x_searchers.keys()

    # loop searchers
    for s in S:
        # loop through time
        for t in range(t_max + 1):
            v_t = [k[1] for k in x_keys if s == k[0] and t == k[2]]
            var_value = [old_x_searchers.get((s, v, t)) for v in v_t]

            # find max value idx at time t
            mx = var_value.index(max(var_value))

            # everything else is zero
            for v in v_t:
                if v == v_t[mx]:
                    x_searchers[(s, v, t)] = 1
                else:
                    x_searchers[(s, v, t)] = 0

    # for debugging only!
    if old_x_searchers != x_searchers:
        print('-X-X-X-X-X-X- Eureka! -X-X-X-X-X-X-')

    return x_searchers


def enforce_sum_1(b_target, t_max):

    old_b_target = b_target
    b_keys = b_target.keys()

    dc = 4

    for t in range(t_max + 1):

        b_values = [round(b_target.get(k), dc) for k in b_keys if t == k[1]]
        v_t = [k for k in b_keys if t == k[1]]

        my_sum = round(sum(b_values), dc)

        # normalize
        for i in range(len(b_values)):
            k = v_t[i]
            vl = round(b_values[i]/my_sum, dc)

            b_target[k] = vl

    return b_target


def query_and_print_variables(md):
    """query variable X to see the optimal path"""

    # save x variable as dictionary with keys (s, v, t)
    x_searchers = {}
    # save beta variable as dictionary with keys (v, t)
    b_target = {}

    for var in md.getVars():
        my_var_name = var.varName
        my_var_value = var.x
        print('%s %g' % (my_var_name, my_var_value))

        if 'x' in my_var_name:
            s = int(my_var_name[2])
            v = int(my_var_name[4])
            t = int(my_var_name[6])

            if my_var_value >= 0.5:
                x_searchers[(s, v, t)] = 1
            else:
                x_searchers[(s, v, t)] = 0

        elif 'beta' in my_var_name:
            # print('%s %g' % (my_var_name, my_var_value))
            # remember b[0] is probability of capture
            v = int(my_var_name[5])
            t = int(my_var_name[7])
            b_target[v, t] = my_var_value

    obj = md.getObjective()
    print(obj.getValue())

    return x_searchers, b_target


# TODO deprecated -- delete (only used for text_sim_fun)
# model
def run_gurobi(graph_file, horizon: int, searchers_info: dict, b0: list, M_target, gamma=0.99):
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

