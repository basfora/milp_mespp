"""Construct MILP model
Functions:
 - Searchers' motion: get vertices for each time step: V(s, t), get previous, current, next vertices
 - Capture events: matrices and checks
 - Target motion: probability matrix, get true starting position
 - Belief: update equation
 - """
import extract_info as ext
import numpy as np


# V(s,t)
def get_vertices_and_steps(G, deadline, searchers):
    """Extract information from the user provided graph and information on searchers
    For each time step, find which vertices each searcher is allowed to be"""

    start = ext.get_searchers_positions(searchers)
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


def get_vertices_and_steps_distributed(G, deadline, searchers, temp_s_path):
    """Extract information from the user provided graph and information on searchers
       For each time step, find which vertices each searcher is allowed to be
       Since this is the distributed version, use info on temporary searchers path (temp_s_path)"""

    start = ext.get_searchers_positions(searchers)
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


# V(t)
def get_next_vertices(g, s: int, v: int, t: int, vertices_t: dict, T_ext: list):
    """Find possible next vertices according to time
     s, v and t refers to the value of each (not index)"""
    v_idx = ext.get_python_idx(v)

    # is at last possible vertex, will be at dummy goal on T + 1 -  don't worry about proximity
    if t == T_ext[-1]:
        return vertices_t.get((s, t + 1))
    # check for physical proximity (neighbors) AND possible vertex at t+1
    elif t < T_ext[-1]:
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


# Capture events: matrices and checks
def get_u_for_capture(searchers: dict, V: list, v: int):
    """Return a list with the all searchers vertex u that could capture the target placed at v"""
    my_list = []
    for s_id in searchers.keys():
        s = searchers[s_id]
        for u in V:
            # get capture matrix
            C = s.get_capture_matrix(u)
            if check_capture(C, v) and u not in my_list:
                my_list.append(u)
    return my_list


def get_capture_s(searchers: dict, s_id, u):
    """get capture matrices from a single searcher"""
    s = searchers[s_id]
    C = s.get_capture_matrix(u)
    return C


def get_all_capture_s(searchers: dict, s_id):
    """get capture matrices from searchers (s_info or searchers class)
    for a single searcher s"""
    # get s
    s = searchers[s_id]

    # s_info or searchers
    if isinstance(s, dict):
        # TODO take out s_info once code is clean
        # old form, extract from s_info
        c_matrices = s.get('c_matrix')
    else:
        # new, extract from class MySearcher
        c_matrices = s.get_all_capture_matrices()

    return c_matrices


def check_capture(C, v):
    """Return true is the target placed at v could be captured"""
    my_value = C[v][0]
    if C[v][0] > 0:
        return True
    else:
        return False


def check_false_negatives(searchers: dict):
    """Set flag on false negatives for true or false, depending on the elements of C"""
    # default
    false_neg = False
    zeta = None
    # get capture matrices from first searcher, first vertex (assume the other will be the same type)
    s = 1
    u = 1
    C = get_capture_s(searchers, s, u)

    # check if it's float
    if C.dtype.type == np.float64:
        false_neg = True
        # TODO allow for diff zetas!!
        if isinstance(searchers[1], dict):
            zeta = searchers[1]['zeta']
        else:
            zeta = searchers[1].zeta

    return false_neg, zeta


def product_capture_matrix(searchers: dict, pi_next_t: dict, n: int):
    """Find and multiply capture matrices for s = 1,...m
    searchers position needs to be dict with key (s, t)"""

    # number of vertices + 1
    nu = n + 1
    C = {}
    prod_C = np.identity(nu)
    # get capture matrices for each searcher that will be at pi(t+1)
    for s_id in searchers.keys():
        s = searchers[s_id]
        # get where the searchers is now
        v = pi_next_t.get(s_id)
        # extract the capture matrix for that vertex
        C[s_id] = s.get_capture_matrix(v)
        # recursive product of capture matrix, from 1...m searchers
        prod_C = np.matmul(prod_C, C[s_id])

    return prod_C


# target motion
def assemble_big_matrix(n: int, Mtarget):
    """Assemble array for belief update equation"""

    if isinstance(Mtarget, list):
        # transform motion matrix in array
        M = np.asarray(Mtarget)
    else:
        M = Mtarget

    # assemble the array
    a = np.array([1])
    b = np.zeros((1, n), dtype=int)
    c = np.zeros((n, 1), dtype=int)
    # extended motion array
    row1 = np.concatenate((a, b), axis=None)
    row2 = np.concatenate((c, M), axis=1)
    # my matrix
    big_M = np.vstack((row1, row2))
    return big_M


def probability_move(M, current_vertex):
    """get moving probabilities for current vertex"""

    # get current vertex id
    v_idx = ext.get_python_idx(current_vertex)
    n = len(M[v_idx])
    prob_move = []
    my_vertices = []
    for col in range(0, n):
        prob_v = M[v_idx][col]
        # if target can move to this vertex, save to list
        if prob_v > 1e-4:
            prob_move.append(prob_v)
            my_vertices.append(col + 1)

    return my_vertices, prob_move


def get_target_true_position(v_target, idx=None):
    """return true position of the target based on the initial vertice distribution
    idx is the index correspondent of the true position of the list v_target"""

    # if there is only one possible vertex, the target is there
    if len(v_target) == 1:
        v_target_true = v_target[0]
    else:
        # if no index for the true vertex was provided, choose randomly
        if idx is None:
            n_vertex = len(v_target)
            prob_uni = (1/n_vertex)
            my_array = np.ones(n_vertex)
            prob_array = prob_uni * my_array
            prob_move = prob_array.tolist()
            v_target_true = ext.sample_vertex(v_target, prob_move)
        # if the index was provided, simply get the position
        else:
            if idx >= len(v_target):
                v_target_true = v_target[-1]
            else:
                v_target_true = v_target[idx]

    return v_target_true


# belief
def belief_update_equation(current_belief: list, big_M: np.ndarray, prod_C: np.ndarray):
    """Update the belief based on Eq (2) of the model:
    b(t+1) = b(t) * big_M * Prod_C"""

    # transform into array for multiplication
    current_b = ext.convert_list_array(current_belief, 'array')

    # use belief update equation
    dummy_matrix = np.matmul(big_M, prod_C)
    new_b = np.matmul(current_b, dummy_matrix)

    # transform to list
    new_belief = ext.convert_list_array(new_b, 'list')

    return new_belief


