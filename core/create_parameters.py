from igraph import *
import numpy as np
from core import extract_info as ext
from classes.class_searcher import MySearcher
import pickle
import random

# ----------------------------------------------------------------------------------------------------------------------
# instance parameters (searchers, target)

def my_target_motion(g, init_vertex: list,  init_prob='uniform',  motion_rule='random'):
    """Create target motion model matrix and initial belief:
    g is the graph
    init_vertex" list with the nodes numbers (starting at 1)
    init_prob: probability for each none in init_vertex.
    If left blank probability will be equal between all vertices in init_vertex """

    b_0 = my_initial_belief(g, init_vertex, init_prob)

    # target motion model - Markovian, random
    M = my_motion_matrix(g, motion_rule)

    return b_0, M


def my_initial_belief(g, init_vertex: list, init_prob='uniform'):
    """ Make initial belief vector based on number of vertices n
    initial vertices (list of 1 or more)
    optional initial probability, =1 when not specified"""

    V_, n = ext.get_idx_vertices(g)
    # create belief vector of zero
    b_0 = np.zeros(n + 1)

    # get number of possible initial vertices
    n_init = len(init_vertex)

    # if the initial_prob is not a list with pre-determined initial probabilities for each possible initial vertex
    # just consider equal probability between possible vertices
    if init_prob is 'uniform':
        prob_init = 1/n_init
    else:
        prob_init = init_prob

    # initial target belief
    b_0[init_vertex] = prob_init

    return list(b_0)


def my_motion_matrix(g,  motion_rule='random'):
    """Define Markovian motion matrix for target
    unless specified, motion is random (uniform) according to neighbor vertices"""
    # TODO in analysis: other types of motions

    V_, n = ext.get_idx_vertices(g)

    M = np.zeros((n, n))

    if motion_rule is 'random':
        for v in V_:
            delta_prime = [v] + g.vs[v]["neighbors"]
            n_nei = len(delta_prime)
            prob_v = 1/n_nei
            M[v, delta_prime] = prob_v
    elif motion_rule == 'static':
            M = np.identity(n)

    return M.tolist()


def my_searchers_info(g, v0, capture_range=0, zeta=None):
    """Give searchers info (dictionary with id number as keys).
    Nested: initial position, capture matrices for each vertex"""
    # get set of searchers based on initial vertex for searchers
    S = ext.get_set_searchers(v0)[0]
    # get graph vertices
    V, n = ext.get_set_vertices(g)

    # check to see if it's a vertex in the graph
    if any(v0) not in V:
        print("Vertex out of range, V = {1, 2...n}")
        return None

    # size of capture matrix
    nu = n + 1
    # create dict
    searchers_info = {}
    for s in S:
        my_aux = {}
        for v in V:
            # loop through vertices to get capture matrices
            C = rule_intercept(v, nu, capture_range, zeta, g)
            my_aux[v] = C
        idx = ext.get_python_idx(s)
        searchers_info.update({s: {'start': v0[idx], 'c_matrix': my_aux, 'zeta': zeta}})
    return searchers_info


def create_searchers(g, v0, capture_range=0, zeta=None):
    """Give searchers info (dictionary with id number as keys).
            Nested: initial position, capture matrices for each vertex"""
    # get set of searchers based on initial vertex for searchers
    S = ext.get_set_searchers(v0)[0]
    # get graph vertices
    V, n = ext.get_set_vertices(g)

    # check to see if it's a vertex in the graph
    if any(v0) not in V:
        print("Vertex out of range, V = {1, 2...n}")
        return None

    # size of capture matrix
    nu = n + 1
    # create dict
    searchers = {}
    for s_id in S:
        my_aux = {}
        for v in V:
            # loop through vertices to get capture matrices
            C = rule_intercept(v, nu, capture_range, zeta, g)
            my_aux[v] = C
        idx = ext.get_python_idx(s_id)
        C_all = my_aux
        s = MySearcher(s_id, v0[idx], C_all, capture_range, zeta)

        searchers[s_id] = s
    return searchers


def rule_intercept(v, nu, capture_range=0, zeta=None, g=None):
    """create C matrix based on the rule of interception
    graph needs to be an input if it's multiple vertices"""

    if zeta is None:
        my_type = 'int32'
        zeta = 0
    else:
        my_type = 'float64'

    # create identity matrix (n+1) x (n+1)
    C = np.identity(nu, my_type)

    # apply rule of interception to create capture matrices
    # same vertex (for now the same for each searcher, but can be different)
    # if integer, zeta = 0 and 1-zeta = 1
    if capture_range == 0:
        C[v][v] = zeta
        C[v][0] = 1 - zeta
    else:
        # list with all the vertices on the graph
        my_row = list(range(1, nu))
        for u in my_row:
            distance = ext.get_node_distance(g, v, u)
            if distance <= capture_range:
                # assemble the capture matrix
                C[u][u] = zeta
                C[u][0] = 1 - zeta
    return C


def check_initial_conditions(v_target: list, v_searchers: list):
    """Check if the searchers and target are in different initial vertices"""
    common_el = set(v_target) - (set(v_target) - set(v_searchers))
    if len(common_el) > 0:
        print("searchers can't be in same vertex as target at t = 0")
        return False
    else:
        return True


def check_reachability(g, capture_range, v_target, v_searchers):
    """Check if the target is within reach of any one of the searchers"""

    init_is_ok = True

    for v in v_searchers:
        for vt in v_target:
            distance = ext.get_node_distance(g, vt, v)
            if distance <= capture_range:
                init_is_ok = False
                return init_is_ok

    return init_is_ok


def init_parameters(g, v_target: list, v_searchers: list, target_motion: str, belief_distribution: str,
                    capture_range=0, zeta=None):

    b_0, M = my_target_motion(g, v_target, belief_distribution, target_motion)
    s_info = my_searchers_info(g, v_searchers, capture_range, zeta)

    return b_0, M, s_info


def searcher_random_pos(v_possible, m: int,  my_seed=None):
    """Choose random vertices for the starting point of the searchers
    positions are given in model indexing (1,2...)"""

    # get set of searchers
    S, m = ext.get_set_searchers(m)

    v_init = []

    if my_seed is None:
        for s in S:
            my_v = int(random.choice(v_possible))
            # append to list
            v_init.append(my_v)
    else:
        v = pick_pseudo_random(v_possible, my_seed, 1)
        for i in S:
            v_init.append(v[0])

    return v_init


def target_random_pos(g, init_possible=1, my_seed=None):
    """Choose random vertices for the starting point of the target
        positions are given in model indexing (1,2...)"""


    # get set of vertices
    V, n = ext.get_set_vertices(g)


    v_target = []

    if my_seed is None:
        v_possible = V
        # randomly pick vertices
        for pos in range(0, init_possible):
            my_v = int(random.choice(v_possible))
            v_target.append(my_v)
            # take out that vertex from list of possible vertices
            v_possible.remove(my_v)
    else:
        v_target = pick_pseudo_random(V, my_seed, init_possible)
        v_possible = [x for x in V if x not in v_target]

    return v_target, v_possible


def target_restricted(g, init_possible=4, my_seed=None):

    # target
    # corners
    v_init_target = [1, 8, 71, 78]
    n_y_target = 3

    v_rtd = random_corners(v_init_target, n_y_target)

    v_target = []

    for i in range(0, init_possible):
        print(v_target)
        my_v = pick_pseudo_random(v_rtd[i], my_seed, 1)
        v_target.append(my_v[0])

    # center
    # searchers
    v_init_searcher = [34]
    n_y_searcher = 4
    v_dict = random_corners(v_init_searcher, n_y_searcher)
    v_possible = v_dict[0]

    # print(v_possible)

    return v_target, v_possible


def random_corners(v_init, n_y, n_columns=10):

    V_rtd = {}

    i = 0
    while i < len(v_init):
        # "groups"
        V_rtd[i] = []
        j = 0
        while j < n_y:
            # rows
            v_it = v_init[i] + j
            v_numbers = list(range(v_it, v_it + (n_y * n_columns), n_columns))
            for el in v_numbers:
                # append
                V_rtd[i].append(el)

            j += 1
            print(V_rtd)
        i += 1

    return V_rtd


def pick_pseudo_random(my_list: list, my_seed: int, qty: int, replace_opt=None):

    # set seed
    np.random.seed(my_seed)

    if replace_opt is None:
        replace_opt = False

    # idx_list = np.random.randint(low=0, high=last_idx, size=qty)
    random_list = np.random.choice(my_list, qty, replace=replace_opt).tolist()

    return random_list


def random_init_pos(g, m: int, init_pos=1, my_seed=None):
    """Choose random vertices for searchers and target
    :param g - graph
    :param m - number of searchers
    :param init_pos - number of possible vertices for the target
    :param my_seed"""

    if my_seed is None:
        my_s_seed, my_t_seed, my_seeds = None, None, None
    else:
        if isinstance(my_seed, dict):
            my_s_seed = my_seed['searcher']
            my_t_seed = my_seed['target']
        else:
            # old code TODO fix this later
            my_s_seed = my_seed + 1000
            my_t_seed = my_seed + 5000

    # returns the vertex for target and searchers in model indexing (1, 2....n)
    # not corners
    v_target, v_possible = target_random_pos(g, init_pos, my_t_seed)
    # uncomment here for target restricted (corners)
    # v_target, v_possible = target_restricted(g, 4, my_t_seed)
    v_searchers = searcher_random_pos(v_possible, m, my_s_seed)

    return v_searchers, v_target


# ----------------------------------------------------------------------------------------------------------------------
# Create graphs

def my_graph(number_vertex: int, ref: str, graph_opt=1, deadline=None, w=None, h=None):
    """Function to create graph according to user inputs"""
    # Graph representing environment
    # create new graph
    g = Graph(directed=False)
    g.add_vertices(number_vertex)

    if graph_opt == 1:
        if deadline is None:
            deadline = 8  # time steps
        g.add_edges([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (4, 5), (5, 6)])
    elif graph_opt == 2:
        if deadline is None:
            deadline = 2
        g.add_edges([(0, 1), (1, 2)])
    elif graph_opt == 3:
        if deadline is None:
            deadline = 3  # time steps
        g.add_edges([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (4, 5), (5, 6), (6, 7)])
    elif graph_opt == 4:
        # create graph from Hollinger 2009 - MUSEUM
        g.add_edges([(0, 55), (1, 55), (2, 55), (3, 55),
                     (55, 56), (55, 54), (55, 34), (55, 33), (55, 35),
                     (54, 32), (54, 31), (54, 30), (54, 53),
                     (53, 29), (53, 28), (53, 27), (53, 26), (53, 39), (53, 52), (53, 38), (53, 37), (53, 36),
                     (52, 57), (52, 25), (52, 50),
                     (50, 51), (50, 49),
                     (25, 24), (24, 43),
                     (57, 56), (57, 4), (57, 5), (57, 6), (57, 40), (57, 41), (57, 43),
                     (49, 48), (48, 43), (48, 59),
                     (43, 42), (43, 44), (43, 22),
                     (44, 7), (44, 8), (44, 9), (44, 21), (44, 23), (44, 45),
                     (21, 45), (21, 22),
                     (46, 45), (45, 10), (46, 11), (46, 12), (46, 13), (46, 47), (46, 20), (47, 14),
                     (58, 47), (58, 15), (58, 59), (58, 18), (58, 19), (59, 17), (59, 16)])

    elif graph_opt == 5:
        g = add_edge_grid(g, w, h)

    elif graph_opt == 7:
        # create graph from Hollinger 2009 - OFFICE
        edges = [(1, 2), (1, 5), (1, 9), (1, 7),
                 (2, 1), (2, 3), (2, 4),
                 (3, 2),
                 (4, 2),
                 (5, 1),
                 (6, 7),
                 (7, 6), (7, 8), (7, 1),
                 (8, 7),
                 (9, 1), (9, 10),
                 (10, 9), (10, 11), (10, 12), (10, 13), (10, 18), (10, 19), (10, 20), (10, 27),
                 (11, 10), (11, 14), (11, 28),
                 (12, 10), (12, 15), (12, 16),
                 (13, 10), (13, 17),
                 (14, 11), (14, 15),
                 (15, 14), (15, 12),
                 (16, 12), (16, 17),
                 (17, 16), (17, 13),
                 (18, 10), (18, 19), (18, 26), (18, 70),
                 (19, 10), (19, 18), (19, 20), (19, 25), (19, 26),
                 (20, 10), (20, 19), (20, 21), (20, 24),
                 (21, 20), (21, 22),
                 (22, 21), (22, 23),
                 (23, 22), (23, 24),
                 (24, 23), (24, 20), (24, 25),
                 (25, 24), (25, 19), (25, 26),
                 (26, 25), (26, 19), (26, 18),
                 (27, 10), (27, 49),
                 (28, 11), (28, 29),
                 (29, 28), (29, 34),
                 (30, 31), (30, 33),
                 (31, 30), (31, 32),
                 (32, 33), (32, 31), (32, 37),
                 (33, 32), (33, 36),
                 (34, 29), (34, 35),
                 (35, 34), (35, 36), (35, 40), (35, 49),
                 (36, 35), (36, 33), (36, 39), (36, 37),
                 (37, 36), (37, 32), (37, 38),
                 (38, 37), (38, 42),
                 (39, 36), (39, 40), (39, 41),
                 (40, 35), (40, 39), (40, 43),
                 (41, 39), (41, 42),
                 (42, 41), (42, 38),
                 (43, 40), (43, 44), (43, 47),
                 (44, 43), (44, 45),
                 (45, 44), (45, 46),
                 (46, 45), (46, 47),
                 (47, 43), (47, 46), (47, 48),
                 (48, 47), (48, 50),
                 (49, 35), (49, 27), (49, 58), (49, 50),
                 (50, 48), (50, 49), (50, 51), (50, 52),
                 (51, 50),
                 (52, 50), (52, 56), (52, 53),
                 (53, 52), (53, 54), (53, 55), (53, 56),
                 (54, 53), (54, 55),
                 (55, 54), (55, 53),
                 (56, 52), (56, 53), (56, 64),
                 (57, 58), (57, 63),
                 (58, 49), (58, 57), (58, 59), (58, 62),
                 (59, 60), (59, 61), (59, 58),
                 (60, 70), (60, 61), (60, 59),
                 (61, 60), (61, 59), (61, 69), (61, 68),
                 (62, 58), (62, 67),
                 (63, 57), (63, 66), (63, 64),
                 (64, 63), (64, 56), (64, 65),
                 (65, 64), (65, 66),
                 (66, 65), (66, 67), (66, 63),
                 (67, 62), (67, 66), (67, 68),
                 (68, 61), (68, 67), (68, 69),
                 (69, 61), (69, 68),
                 (70, 18), (70, 60)]

        edges = map(lambda x: (x[0] - 1, x[1] - 1), edges)
        g.add_edges(edges)

        g.simplify(multiple=True, loops=False)

    V_, n = ext.get_idx_vertices(g)
    V = ext.get_set_vertices(g)[0]

    if deadline is None:
        deadline = 10

    # label starting at 1
    g.vs["label"] = V

    # find shortest path length between any two vertices
    short_paths = g.shortest_paths_dijkstra()
    g["path_len"] = short_paths

    # find neighbors to each vertex
    for v in V_:
        nei = g.neighbors(v)
        g.vs[v]["neighbors"] = nei

    # name graph
    g["name"] = ref
    return g, deadline


def create_grid_graph(w: int, h: int):
    """wxh graph"""
    n_vertex = w*h
    ref = "G" + str(n_vertex) + 'V' + '_grid'
    graph_opt = 5
    deadline = None
    g, deadline = my_graph(n_vertex, ref, graph_opt, deadline, w, h)
    save_graph(g, ref)
    plot_simple_graph(g)


def add_edge_grid(g, w, h):

    n = len(g.vs)
    vertices = list(range(0, n))

    for vertex in vertices:
        # not the first column
        if vertex % w != 0:
            g.add_edge(vertex, vertex - 1)
        # not the last row
        if vertex < w*(h - 1):
            g.add_edge(vertex, vertex + w)

    return g


def save_and_plot_graph(g, ref: str, v_searchers: list, v_target: list):
    save_graph(g, ref)
    plot_graph(g, ref, v_searchers, v_target)


def save_graph(g, ref):
    """Save created graph as pickle file"""
    graph_name = str(ref) + ".p"
    file_path = ext.get_whole_path(graph_name, 'graphs')
    pickle.dump(g, open(file_path, "wb"))


def plot_graph(g, ref, v_searchers: list, v_target: list):
    """Plot my graph and show me where the searchers started"""

    g.vs["color"] = "green"
    start_searcher = ext.get_python_idx(v_searchers)
    start_target = ext.get_python_idx(v_target)
    for st in start_searcher:
        g.vs[st]["color"] = "blue"
    for st in start_target:
        g.vs[st]["color"] = "red"
    layout = g.layout("kk")
    name_file = "G_" + ref + ".pdf"
    plot(g, name_file, layout=layout, bbox=(300, 300), margin=20)


def plot_simple_graph(g, my_layout='grid'):
    graph_file = g["name"] + ".pdf"
    name_file = ext.get_whole_path(graph_file, 'graphs')
    g.vs["color"] = "gray"
    plot(g, name_file, layout=my_layout)


def create_papers_graph():
    """MESPP paper, Fig 2, first graph (MUSEUM)"""
    n_vertex = 60
    ref = 'G' + str(n_vertex) + 'V'
    graph_opt = 4
    deadline = 10
    g, deadline = my_graph(n_vertex, ref, graph_opt, deadline)
    save_graph(g, ref)

    return g


def create_office_graph():
    """MESPP paper, Fig 2, second graph (OFFICE)"""
    n_vertex = 70
    ref = 'G' + str(n_vertex) + 'V' + '_OFFICE'
    graph_opt = 7
    deadline = 10
    g, deadline = my_graph(n_vertex, ref, graph_opt, deadline)
    save_graph(g, ref)
    plot_simple_graph(g, 'kk')

    return g


if __name__ == "__main__":
    create_grid_graph(3, 3)
