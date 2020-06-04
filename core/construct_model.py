import numpy as np
from core import extract_info as ext
from gurobipy import *


def get_var(my_vars: dict, name: str):
    """Retrieve variable from my_var dictionary according to name"""
    desired_var = my_vars.get(name)
    if desired_var is not None:
        return desired_var
    else:
        var_names = 'x, y, alpha, beta, zeta, psi'
        print('No variable with this name, current model accepts only:' + var_names)
        return None


def init_dict_variables(n_var: int):
    """create empty variables pertinent to the model"""
    # pack things
    my_vars = []
    for i in range(0, n_var):
        my_vars.append({})

    return my_vars


def get_vertices_and_steps(G, deadline, searchers_info):
    """Extract information from the user provided graph and information on searchers
    For each time step, find which vertices each searcher is allowed to be"""

    start = ext.get_start_set(searchers_info)
    # S_ and Tau
    S, m = ext.get_set_searchers(start)
    Tau = ext.get_set_time(deadline)
    # get vertices
    V, n = ext.get_set_vertices(G)

    # shortest path lengths
    spl = ext.get_length_short_paths(G)

    # find V^tau(t) = {v in V} --> possible v(t) in optimal solution
    vertices_t = {}
    # find V^tau(v) = {t in Tau} --> possible t(v) in optimal solution
    times_v = {}

    start_idx = ext.get_python_idx(start)

    for s in S:
        s_idx = ext.get_python_idx(s)
        # initial position
        vertices_t[s, 0] = [start[s_idx]]

        # starting at t = 1 (idx = 1), it begins to move
        for t in Tau:
            vertices_t[s, t] = []
            # find the nodes that are within t of distance (thus can be reached at t)
            for v in V:
                v_idx = ext.get_python_idx(v)
                dummy_var = spl[start_idx[s_idx]][v_idx]
                if dummy_var <= t:
                    vertices_t[s, t].append(v)

        # find times allowed for each vertex
        for v in V:
            times_v[s, v] = []
            if v == start[s_idx]:
                times_v[s, v] = [0]

            for t in Tau:
                if v in vertices_t[s, t]:
                    times_v[s, v].append(t)

        # find dummy goal vertex and T + 1
        v_g = ext.get_label_dummy_goal(V)
        t_g = ext.get_last_t(Tau)
        # append dummy goal
        vertices_t[s, t_g] = [v_g]   # at T + 1, searcher is at dummy goal vertex
        times_v[s, v_g] = [t_g]      # the only time allowed for the dummy goal vertex is T + 1

    return start, vertices_t, times_v


def get_vertices_and_steps_distributed(G, deadline, searchers_info, temp_s_path):
    """Extract information from the user provided graph and information on searchers
       For each time step, find which vertices each searcher is allowed to be
       Since this is the distributed version, use info on temporary searchers path (temp_s_path)"""

    start = ext.get_start_set(searchers_info)
    # S_ and Tau
    S, m = ext.get_set_searchers(start)
    Tau = ext.get_set_time(deadline)
    # get vertices
    V, n = ext.get_set_vertices(G)

    # shortest path lengths
    spl = ext.get_length_short_paths(G)

    # find V^tau(t) = {v in V} --> possible v(t) in optimal solution
    vertices_t = {}
    # find V^tau(v) = {t in Tau} --> possible t(v) in optimal solution
    times_v = {}

    for s in S:
        # initial position
        vertices_t[s, 0] = [temp_s_path[(s, 0)]]
        st_idx = ext.get_python_idx(vertices_t.get((s, 0))[0])

        # starting at t = 1 (idx = 1), it begins to move
        for t in Tau:
            vertices_t[s, t] = []

            if s == temp_s_path['current_searcher']:
                # if it's planning for this searcher, consider all possibilities
                # find the nodes that are within t of distance (thus can be reached at t)

                for v in V:
                    v_idx = ext.get_python_idx(v)
                    dummy_var = spl[st_idx][v_idx]
                    if dummy_var <= t:
                        vertices_t[s, t].append(v)
            else:

                # if is not the planning searcher, just use the info on the temporary path
                # either the start vertex of the pre-computed path
                v = temp_s_path[s, t]
                vertices_t[s, t].append(v)

        # find times allowed for each vertex
        for v in V:
            times_v[s, v] = []
            if v == vertices_t[s, 0][0]:
                times_v[s, v] = [0]

            for t in Tau:
                if v in vertices_t[s, t]:
                    times_v[s, v].append(t)

        # find dummy goal vertex and T + 1
        v_g = ext.get_label_dummy_goal(V)
        t_g = ext.get_last_t(Tau)
        # append dummy goal
        vertices_t[s, t_g] = [v_g]  # at T + 1, searcher is at dummy goal vertex
        times_v[s, v_g] = [t_g]  # the only time allowed for the dummy goal vertex is T + 1

    return start, vertices_t, times_v


def get_next_vertices(g, s: int, v: int, t: int, vertices_t: dict, Tau_ext: list):
    """Find possible next vertices according to time
     s, v and t refers to the value of each (not index)"""
    v_idx = ext.get_python_idx(v)

    # is at last possible vertex, will be at dummy goal on T + 1 -  don't worry about proximity
    if t == Tau_ext[-1]:
        return vertices_t.get((s, t + 1))
    # check for physical proximity (neighbors) AND possible vertex at t+1
    elif t < Tau_ext[-1]:
        v_t_next = []
        v_nei = g.vs[v_idx]["neighbors"] + [v_idx]
        # get labels
        for neighbor in v_nei:
            u = ext.get_label_name(neighbor)
            if u in vertices_t.get((s, t + 1)):
                v_t_next.append(u)
        return v_t_next
    else:
        return None


def get_current_vertices(t: int, vertices_t: dict, S: list):
    my_vertices = {}
    for s in S:
        my_vertices[s] = vertices_t.get((s, t))
    if my_vertices is not None:
        return my_vertices
    else:
        return None


def get_previous_vertices(g, s: int, v: int, t: int, vertices_t: dict):
    """Find possible previous vertices according to time """
    v_idx = ext.get_python_idx(v)
    # do not have previous data (before start)
    if t == 0:
        return None
    # check for physical proximity (neighbors) AND possible vertex at t+1
    else:
        v_t_previous = []
        v_nei = g.vs[v_idx]["neighbors"] + [v_idx]
        for neighbor in v_nei:
            u = ext.get_label_name(neighbor)
            if u in vertices_t.get((s, t-1)):
                v_t_previous.append(u)
        return v_t_previous


def add_searcher_variables(md, g, start: list, vertices_t: dict, deadline: int):
    """Add variables related to the searchers on the model:
    searchers location at each time step, X
    searchers movement from t to t + 1, Y"""

    [X, Y] = init_dict_variables(2)

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
                my_next_v = get_next_vertices(g, s, v, t, vertices_t, Tau_ext)
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


def add_target_variables(md, g, deadline: int, s_info=None):
    """Add variables related to the target and capture events:
    belief variable, B
    interception-related variables: belief vector composition, beta
    belief evolution, alpha
    capture event, zeta and psi
    """
    # TODO test this

    V = ext.get_set_vertices(g)[0]

    Tau_ext = ext.get_set_ext_time(deadline)
    Tau = ext.get_set_time(deadline)

    V_ext = ext.get_set_ext_vertices(g)

    var_for_test = {}
    list_beta_name = []
    list_alpha_name = []
    list_delta_name = []

    [beta, beta_s, alpha, psi, delta] = init_dict_variables(5)

    if s_info is not None:
        false_neg, zeta = check_false_negatives(s_info)
        S = ext.get_set_searchers(s_info)[0]
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
                my_next_v = get_next_vertices(g, s, u, t, vertices_t, Tau_ext)
                my_previous_v = get_previous_vertices(g, s, u, t, vertices_t)
                if my_next_v is not None:
                    # (Eq. 9) searcher can only move to: i in delta_prime(v) AND V^tau(t+1)
                    # sum == 1 if searcher is at u, sum == zero if searcher is not at u (depends on X[s, u, t])
                    md.addConstr(quicksum(Y[s, u, i, t] for i in my_next_v) == X[s, u, t])

                    if my_previous_v is not None:
                        # (Eq. 8) searcher can only move to v from j in delta_prime(v) AND V^tau(t-1)
                        md.addConstr(quicksum(Y[s, i, u, t - 1] for i in my_previous_v) == X[s, u, t])
                        # md.addConstr(quicksum(Y[s, i, u, t - 1] for i in my_previous_v) ==
                                     # quicksum(Y[s, u, i, t] for i in my_next_v))


def add_capture_constraints(md, g, my_vars: dict, searchers_info: dict, vertices_t, b0: list, M: list, deadline: int):
    """Define constraints about belief and capture events
    :param vertices_t:
    :param md:
    :param g:
    :param my_vars:
    :param searchers_info:
    :param b0
    :param deadline
    """

    # capture-related variables
    beta = get_var(my_vars, 'beta')
    alpha = get_var(my_vars, 'alpha')
    psi = get_var(my_vars, 'psi')

    false_neg, zeta = check_false_negatives(searchers_info)

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
    S, m = ext.get_set_searchers(searchers_info)

    Tau = ext.get_set_time(deadline)
    V_ext = ext.get_set_ext_vertices(g)

    # initial belief (user input), t = 0 (Eq. 13)
    for i in V_ext:
        md.addConstr(beta[i, 0] == b0[i])

    for t in Tau:
        # this is a dictionary
        my_vertices = get_current_vertices(t, vertices_t, S)
        for v in V:
            # v_idx = ext.get_python_idx(v)
            # take Markovian model into account (Eq. 14)
            # NOTE M matrix is accessed by python indexing
            md.addConstr(alpha[v, t] == quicksum(M[u - 1][v - 1] * beta[u, t - 1] for u in V))

            # find searchers position that could capture the target while it is in v
            list_u_capture = get_u_for_capture(searchers_info, V, v)
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


def get_u_for_capture(searchers_info: dict, V: list,  v: int):
    """Return a list with the searchers vertex u that could capture the target placed at v"""
    my_list = []
    for s in searchers_info.keys():
        for u in V:
            # get capture matrix
            C = get_capture_matrix(searchers_info, s, u)
            if check_capture(C, v) and u not in my_list:
                my_list.append(u)
    return my_list


def get_capture_matrix(searchers_info: dict, s, u):
    """get capture matrices from searchers_info"""
    c_matrices = get_all_capture_matrices(searchers_info, s)
    C = c_matrices.get(u)
    return C


def get_all_capture_matrices(searchers_info: dict, s):
    """get capture matrices from searchers_info"""
    my_aux_dict = searchers_info.get(s)
    c_matrices = my_aux_dict.get('c_matrix')
    return c_matrices


def check_capture(C, v):
    """Return true is the target placed at v could be captured"""
    my_value = C[v][0]
    if C[v][0] > 0:
        return True
    else:
        return False


def check_false_negatives(s_info: dict):
    """Set flag on false negatives for true or false, depending on the elements of C"""
    # default
    false_neg = False
    zeta = None
    # get capture matrices from first searcher, first vertex (assume the other will be the same type)
    s = 1
    u = 1
    C = get_capture_matrix(s_info, s, u)

    # check if it's float
    if C.dtype.type == np.float64:
        false_neg = True
        zeta = s_info[1]['zeta']

    return false_neg, zeta