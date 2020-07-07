"""Create parameters for the problem instance"""

from core import extract_info as ext
from classes.searcher import MySearcher
from classes.belief import MyBelief
from classes.target import MyTarget
from classes.solver_data import MySolverData
from classes.inputs import MyInputs

# external packages
from igraph import *
import numpy as np
import pickle
import random


# init inputs class
def define_specs():
    """Initialize pre-set parameters
    If needed, change parameters here using MyInputs() class functions
    Return: specs"""

    specs = MyInputs()
    # -------------------
    # call here MyInputs() functions to change default specs
    # -------------------

    return specs


# init classes from specs (exp_inputs)
def create_solver_data(specs):
    """Initialize solver data class from specs (exp_inputs)"""
    g = specs.graph

    # planning stuff
    deadline = specs.deadline
    theta = specs.theta
    horizon = specs.horizon
    solver_type = specs.solver_type
    timeout = specs.timeout

    solver_data = MySolverData(horizon, deadline, theta, g, solver_type, timeout)

    return solver_data


def create_belief(specs):
    """Initialize belief
    either from distribution (uniform, split among random vertices)
    or from given b0
    """

    if specs.b0 is None:
        g = specs.graph
        # type of distribution (default: uniform)
        type_distribution = specs.belief_distribution
        v_list = placement_list(specs, 't')
        # create b_0 (list with probabilities, b[0] = 0)
        b_0 = set_initial_belief(g, v_list, type_distribution)
    else:
        # user-defined initial belief (with b_c)
        b_0 = specs.b0

    # save on specs
    v_list = v_list_from_belief(b_0)
    specs.set_set_start_target_v_list(v_list)

    # create belief obj
    belief = MyBelief(b_0)

    return belief


def create_searchers(specs):

    # unpack from specs
    g = specs.graph
    capture_range = specs.capture_range
    zeta = specs.zeta
    m = specs.size_team

    if specs.start_searcher_v is None:
        # if initial position was not defined by user
        v_list = placement_list(specs, 's')
        if specs.searcher_together:
            v_list = searchers_start_together(m, v_list)

        # len(v0) = m
        v0 = v_list
        specs.set_start_searcher(v0)
    else:
        # if it was, use that
        v0 = specs.start_searcher_v

    # set of searchers S = {1,..m}
    S = ext.get_set_searchers(m)[0]
    # create dict
    searchers = {}
    for s_id in S:
        v = ext.get_v0_s(v0, s_id)
        cap = ext.get_capture_range_s(capture_range, s_id)
        zeta_s = ext.get_zeta_s(zeta, s_id)

        # create each searcher
        s = MySearcher(s_id, v, g, cap, zeta_s)

        # store in dictionary
        searchers[s_id] = s
    return searchers


def create_target(specs):
    """Needs to be called AFTER belief class"""
    # TODO expand for starting target in specific position
    # for now: if wants to start target in place, set specs.start_target_v_list,
    # with the first item being the true position

    # unpack from target
    motion_rule = specs.target_motion
    g = specs.graph
    my_seed = specs.target_seed

    v_list = specs.start_target_v_list
    true_position = specs.start_target_true

    # generate motion matrix M
    motion_matrix = my_motion_matrix(g, motion_rule)

    target = MyTarget(v_list, motion_matrix, true_position, my_seed)

    return target


# ----------------------------------------------------------------------------------------------------------------------
# instance parameters
def draw_v_random(g_or_n, q=1, my_seed=None):
    """Choose possible random vertices for the starting point of the target
    positions are given in model indexing (1,2...)
    :param g_or_n : graph
    :param q : number of possible vertices
    :param my_seed : random seed generator (optional)
    return: v_target --> possible initial vertices of the target
     v_left: target from the graph that are 'free' """
    # get set of vertices
    V, n = ext.get_set_vertices(g_or_n)

    v_target = []

    if my_seed is None:
        v_left = V

        # randomly pick vertices
        for pos in range(0, q):
            my_v = int(random.choice(v_left))
            v_target.append(my_v)

            # take out that vertex from list of possible vertices
            v_left.remove(my_v)
    else:
        v_target = pick_pseudo_random(V, my_seed, q)
        v_left = ext.get_v_left(n, v_target)

    return v_target, v_left


def pick_pseudo_random(my_list: list, my_seed: int, qty: int, replace_opt=None):

    # set seed
    np.random.seed(my_seed)

    if replace_opt is None:
        replace_opt = False

    # idx_list = np.random.randint(low=0, high=last_idx, size=qty)
    random_list = np.random.choice(my_list, qty, replace=replace_opt).tolist()

    return random_list


def placement_list(specs, op='s'):
    """Make sure belief and searchers' start vertices are far away (out of reach)
    so that b_c(0) = 0
    :param specs : inputs
    :param op : 's' to place searchers, 't' to get list of vertices for belief """

    g = specs.graph

    # placing searchers or belief?
    if op == 's':
        v_input = specs.start_target_v_list
        my_seed = specs.searcher_seed
        # check if searchers are starting together
        if specs.searcher_together:
            q = 1
        else:
            q = specs.size_team
    else:
        v_input = specs.start_searcher_v
        # quantity of possible nodes (probability > 0)
        q = specs.qty_possible_nodes
        # seed
        my_seed = specs.target_seed

    if v_input is None:
        # draw q random nodes
        v_list, v_left = draw_v_random(g, q, my_seed)
    else:
        # check if it's out of reach at t = 0
        v_taken = v_input
        v_list = []
        out_of_reach = False
        # keep drawing until is out of reach
        while out_of_reach is False:
            v_list = draw_v_random(g, q, my_seed)
            specs.change_seed(my_seed, 't')
            out_of_reach = check_reachability(g, specs.capture_range, v_list, v_taken)
            my_seed += 500

    # check for bug (not a vertex in graph)
    V = ext.get_set_vertices(g)[0]
    if any(v_list) not in V:
        print("Vertex out of range, V = {1, 2...n}")
        exit()

    return v_list


# instance parameters (belief)
def set_initial_belief(g_or_n, v_list: list, type_distribution='uniform'):
    """ Make initial belief vector (assign probabilities to list of vertices) based on
    :param g_or_n : graph or n, to obtain number of total vertices (n)
    :param v_list : list of vertices # [1, 3...] to be assigned non zero probability
    :param type_distribution : type of distribution, default is uniform """

    n = ext.get_set_vertices(g_or_n)[1]

    # create belief vector of zeros
    b_0 = np.zeros(n + 1)

    # get number of possible initial vertices
    q = len(v_list)

    # if the initial_prob is not a list with pre-determined initial probabilities
    # for each possible initial vertex
    # just consider equal probability between possible vertices
    if type_distribution is 'uniform':
        prob_init = 1/q
    else:
        prob_init = type_distribution

    # initial target belief (vertices in v_list will change value, others remain zero)
    # b[0] = 0
    b_0[v_list] = prob_init

    return list(b_0)


# instance parameters (target)
def my_target_motion(g, init_vertex: list,  init_prob='uniform',  motion_rule='random'):
    """Create target motion model matrix and initial belief:
    g is the graph
    init_vertex" list with the nodes numbers (starting at 1)
    init_prob: probability for each none in init_vertex.
    If left blank probability will be equal between all vertices in init_vertex """

    b_0 = set_initial_belief(g, init_vertex, init_prob)

    # target motion model - Markovian, random
    M = my_motion_matrix(g, motion_rule)

    return b_0, M


def my_motion_matrix(g,  motion_rule='random'):
    """Define Markovian motion matrix for target
    unless specified, motion is random (uniform) according to neighbor vertices"""

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

    my_M = M.tolist()
    return my_M


def v_list_from_belief(b0: list):

    v_list = []
    b_v = b0[1:]

    v = 0
    for el in b_v:
        v += 1
        if el > 0.0001:
            v_list.append(v)

    return v_list


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


def list_random_q(turns=200):

    list_qty = []

    for i in range(0, turns):
        random_n = np.random.randint(low=2, high=15)
        list_qty.append(random_n)

    return list_qty


# instance parameters (searchers)
def searchers_start_together(m: int, v):
    """Place all searchers are one vertex
    :param m : number of searchers
    :param v : integer or list"""

    if isinstance(v, int):
        my_v = v
    else:
        my_v = v[0]

    v_searchers = []
    for i in range(m):
        v_searchers.append(my_v)

    return v_searchers


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
            C = MySearcher.rule_intercept(v, nu, capture_range, zeta, g)
            my_aux[v] = C
        idx = ext.get_python_idx(s)
        searchers_info.update({s: {'start': v0[idx], 'c_matrix': my_aux, 'zeta': zeta}})
    return searchers_info


def create_my_searchers(g, v0: list, capture_range=0, zeta=None):
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

    cap_s, zeta_s = 0, None
    # create dict
    searchers = {}
    for s_id in S:
        idx = ext.get_python_idx(s_id)

        v_s = v0[idx]

        if isinstance(capture_range, int):
            cap_s = capture_range
        elif isinstance(capture_range, list):
            cap_s = capture_range[idx]

        if zeta is not None:
            if isinstance(zeta, list):
                zeta_s = zeta[idx]
            elif isinstance(zeta, float):
                zeta_s = zeta

        # create each searcher
        s = MySearcher(s_id, v_s, g, cap_s, zeta_s)
        # store in dictionary
        searchers[s_id] = s
    return searchers


def check_same_vertex(v_target: list, v_searchers: list):
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


# ----------------------------------------------------------------------------------------------------------------------
# Create graphs
def my_graph(number_vertex: int, ref: str, graph_opt=1, w=None, h=None):
    """Function to create graph according to user inputs"""
    # Graph representing environment
    # create new graph
    g = Graph(directed=False)
    g.add_vertices(number_vertex)

    if graph_opt == 1:
        g.add_edges([(0, 1), (0, 2), (1, 3), (1, 4), (2, 4), (4, 5), (5, 6)])
    elif graph_opt == 2:
        g.add_edges([(0, 1), (1, 2)])
    elif graph_opt == 3:
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
    return g


def create_grid_graph(w: int, h: int):
    """wxh graph"""
    n_vertex = w*h
    ref = "G" + str(n_vertex) + 'V' + '_grid'
    graph_opt = 5
    deadline = None
    g = my_graph(n_vertex, ref, graph_opt, w, h)
    save_graph(g, ref)
    plot_simple_graph(g)
    save_pdf(g, ref)


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
    plot_start_config(g, ref, v_searchers, v_target)
    save_pdf(g, ref)


def save_graph(g, ref):
    """Save created graph as pickle file"""
    graph_name = str(ref) + ".p"
    file_path = ext.get_whole_path(graph_name, 'graphs')
    pickle.dump(g, open(file_path, "wb"))


def plot_start_config(g, ref, v_searchers: list, v_target: list):
    """Plot my graph and show me where the searchers started"""

    g.vs["color"] = "green"
    start_searcher = ext.get_python_idx(v_searchers)
    start_target = ext.get_python_idx(v_target)
    for st in start_searcher:
        g.vs[st]["color"] = "blue"
    for st in start_target:
        g.vs[st]["color"] = "red"
    layout = g.layout("kk")
    name_file = "G" + ref + ".pdf"
    plot(g, name_file, layout=layout, bbox=(300, 300), margin=20)


def save_pdf(g, ref):
    """Plot my graph and show me where the searchers started"""

    g.vs["color"] = "gray"
    my_layout = g.layout("kk")
    name_file = ref + ".pdf"
    file_path = ext.get_whole_path(name_file, 'graphs')

    plot(g, file_path, layout=my_layout, bbox=(300, 300), margin=20)


def plot_simple_graph(g, my_layout='grid'):
    graph_file = g["name"] + ".pdf"
    name_file = ext.get_whole_path(graph_file, 'graphs')
    g.vs["color"] = "gray"
    plot(g, name_file, layout=my_layout)


def create_office_graph():
    """MESPP paper, Fig 2, first graph (MUSEUM)"""
    n_vertex = 60
    ref = 'G' + str(n_vertex) + 'V'
    graph_opt = 4
    g = my_graph(n_vertex, ref, graph_opt)
    save_graph(g, ref)

    return g


def create_museum_graph():
    """MESPP paper, Fig 2, second graph (OFFICE)"""
    n_vertex = 70
    ref = 'G' + str(n_vertex) + 'V' + '_OFFICE'
    graph_opt = 7
    g = my_graph(n_vertex, ref, graph_opt)
    save_graph(g, ref)
    plot_simple_graph(g, 'kk')

    return g


def create_graph_test():
    n_vertex = 7
    ref = 'G' + str(n_vertex) + 'V'
    graph_opt = 1
    g = my_graph(n_vertex, ref, graph_opt)
    save_graph(g, ref)
    save_pdf(g, ref)


if __name__ == "__main__":
    create_graph_test()


