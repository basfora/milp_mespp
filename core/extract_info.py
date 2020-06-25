"""Basic functions used throughout the project:
 - get sets
 - get indexes
 - get labels
 - get labels
 - get last
 - names according to notation code
 - extract data
 - sampling from distributions
 - graph related info
 - retrieve specific graphs
 - convert data format
"""

import datetime
import os
import pickle
import random
import numpy as np


# --------------------
# get sets
# --------------------
def get_set_searchers(info_input):
    """Return set of searchers,
     S = {1,...m}"""
    if isinstance(info_input, list):
        m = len(info_input)
        S = list(range(1,  m + 1))
    elif isinstance(info_input, dict):
        S = list(info_input.keys())
        m = len(S)
    elif isinstance(info_input, int):
        m = info_input
        S = list(range(1, m + 1))
    else:
        print("Wrong type of data")
        m = None
        S = None
    return S, m


def get_m_from_xs(x_searchers: dict):
    """Get number of searchers from triple tuple (s, v, t)
    UT-OK"""

    x_keys = x_searchers.keys()

    # list of s
    s_in_keys = [k[0] for k in x_keys]
    # max s
    m = max(s_in_keys)

    return m


def get_set_time(deadline):
    """Return time-step set:
     Tau = {1,2...T},  T = deadline"""
    Tau = list(range(1, deadline + 1))
    return Tau


def get_set_ext_time(time_input):
    """Get time set with deadline + 1 {0, 1, 2, ... deadline}
    (next step at final of simulation for dummy goal placement)"""
    if isinstance(time_input, int):
        Tau_ext = get_idx_time(time_input + 1)
        return Tau_ext
    elif isinstance(time_input, list):
        Tau_ext = time_input + [time_input[-1] + 1]
        return Tau_ext
    else:
        print('Wrong type of input, accepts integer or list')
        return None


def get_set_vertices(g):
    """Return set of graph vertices,
    V = {1, 2,...n}"""
    n = len(g.vs)
    V = list(range(1, n+1))
    return V, n


def get_set_ext_vertices(g):
    """Get indexes for the belief vector:"
    0: capture
    1,...n: vertices"""

    n = len(g.vs)
    V = list(range(1, n + 1))
    V_ext = [0] + V
    return V_ext


def get_start_set(searchers: dict):

    start = []

    # TODO take out s_info once code is clean
    # s_info or searchers
    for s_id in searchers.keys():
        s = searchers[s_id]

        # old form, extract from s_info
        if isinstance(s, dict):
            start_s = s["start"]
        else:
            start_s = s.current_pos
        start.append(start_s)

    return start


def get_sets_only(searcher_vector: list, deadline: int, g):

    S, m = get_set_searchers(searcher_vector)
    Tau = get_set_ext_time(deadline)
    V, n = get_set_vertices(g)
    return S, V, n, Tau


def get_sets_and_ranges(g, m: int, horizon: int):

    S, m = get_set_searchers(m)
    Tau = get_set_ext_time(horizon)
    V, n = get_set_vertices(g)
    return S, V, Tau, n, m


# ---------------------
# get indexes
# ---------------------
def get_idx_searchers(info_vector):
    """Return index of searchers (python style)
         S_IDX = {0,...m-1}"""
    m = len(info_vector)
    S_ = list(range(0, m))
    return S_, m


def get_idx_time(deadline):
    """Return time-step indexing:
     Tau = {0,1...T-1},  T = deadline"""
    Tau_ = list(range(0, deadline))
    return Tau_


def get_idx_vertices(g):
    """Return index of graph vertices,
    V = {0, 2,...n-1}"""
    n = len(g.vs)
    V_ = list(range(0, n))
    return V_, n


def get_python_idx(i):
    """Return i-1 for python indexing of array, list etc"""
    if isinstance(i, list):
        new_list = []
        for el in i:
            new_list.append(el-1)
        return new_list
    else:
        return i-1


def get_start_idx(searchers_info):
    """Return list of searchers start position
    start indexing follow python style [0,1..]"""
    start = []
    for k in searchers_info.keys():
        dummy_var = searchers_info[k]['start']
        my_var = dummy_var - 1
        start.append(my_var)
    return start


def get_idx_dummy_goal(vertex_input):
    """Get the index of the fictitious dummy goal (last vertex + 1)"""

    if isinstance(vertex_input, int):
        v_g = vertex_input + 1
        return v_g
    elif isinstance(vertex_input, list):
        v_g = vertex_input[-1] + 1
        return v_g
    else:
        print('Wrong type of input, accepts integer or list')
        return None


# ---------------------
# get labels
# ---------------------
def get_label_name(i):
    """Return label idx for python array, list etc"""
    if isinstance(i, list):
        new_list = []
        for el in i:
            new_list.append(el+1)
        return new_list
    elif isinstance(i, int):
        return i+1
    elif isinstance(i, dict):
        my_new_dict = {}
        for s in i.keys():
            new_list = []
            my_vertices = i.get(s)
            for v in my_vertices:
                new_list.append(v + 1)
            my_new_dict[s] = new_list
        return my_new_dict
    else:
        return None


def get_start_vertex(searchers_info):
    """Return list of searchers start position
    start indexing follow model style [1,2..]"""
    start = []
    for k in searchers_info.keys():
        dummy_var = searchers_info[k]['start']
        my_var = dummy_var
        start.append(my_var)
    return start


def get_label_dummy_goal(vertex_input):
    """Get the index of the fictitious dummy goal (last vertex + 1)"""

    if isinstance(vertex_input, int):
        v_g = vertex_input + 1
        return v_g
    elif isinstance(vertex_input, list):
        v_g = vertex_input[-1] + 1
        return v_g
    else:
        print('Wrong type of input, accepts integer or list')
        return None


# ---------------------
# get last
# ---------------------
def get_last_t(t_input):
    """Get the index of the fictitious dummy goal (last vertex + 1)"""

    if isinstance(t_input, int):
        t_g = t_input + 1
        return t_g
    elif isinstance(t_input, list):
        t_g = t_input[-1] + 1
        return t_g
    else:
        print('Wrong type of input, accepts integer or list')
        return None


def get_last_key(my_dict: dict):
    """get the last key from a given dictionary"""
    # get last time
    key_array = list(my_dict.keys())
    key_last = key_array[-1]
    return key_last


def get_last_info(my_dict: dict):
    """Get last key and value on dictiornaries of type {key: value}"""
    # get last time (or last key)
    current_key = get_last_key(my_dict)
    # get current belief
    current_value = my_dict.get(current_key)

    return current_key, current_value


# -------------------------------------
# names according to notation code
# -------------------------------------
def add_0_str(number: int):
    """add 0 to the beginning to the string if integer is less than 10"""

    if number < 10:
        # add a zero
        n_str = '0' + str(number)
    else:
        n_str = str(number)
    return n_str


def add_00_str(number: int):
    """add 0 to the beginning to the string if integer is less than 10"""

    if number < 10:
        # add a zero
        n_str = '00' + str(number)
    elif number < 100:
        n_str = '0' + str(number)
    else:
        n_str = str(number)
    return n_str


def get_name_code_folder(today_run: int, m_searcher, g, solver_type='central', zeta=None, capture_range='same_vertex',
                         horizon=None, day_start=None):
    """Give back the name of the folder
    which follows:
    today's date + _ + today's run
    if single digit, add zero in front"""

    # get todays date
    d = datetime.datetime.today()
    m = d.month
    if day_start is None:
        day = d.day
    else:
        day = day_start

    day_str = add_0_str(day)
    m_str = add_0_str(m)
    r_str = add_00_str(today_run)

    # code
    # solver type
    solver = ""
    if solver_type == 'central':
        solver = 'C'
    elif solver_type == 'distributed':
        solver = 'D'

    if horizon is None:
        H = ""
    else:
        H = 'H' + str(horizon)

    # number of searchers
    number_searchers = 'S' + str(m_searcher)

    # graph used
    graph_number = ""
    if g["name"] == 'G60V.p':
        graph_number = 1
    elif g["name"] == 'G64V_grid.p':
        graph_number = 2
    elif g["name"] == 'G256V_grid.p':
        graph_number = 3

    graph_used = 'G' + str(graph_number)

    # false negative or not
    if zeta is None:
        meas_type = 'TN'
    else:
        meas_type = 'FN'

    # capture range
    if capture_range == 0:
        cap_range = 'SV'
    else:
        cap_range = 'MV'

    name_code = solver + H + number_searchers + graph_used + meas_type + cap_range

    # assemble folder name
    name_folder = name_code + "_" + m_str + day_str + "_" + r_str
    whole_path = get_whole_path(name_folder, 'data')
    return name_folder, whole_path


def get_codename(exp_input, folder_parent='data'):
    """Give back the name of the folder
    which follows:
    today's date + _ + today's run
    if single digit, add zero in front"""

    # get todays date
    d = datetime.datetime.today()
    m = d.month
    day = exp_input.start_day

    today_run = exp_input.today_run

    day_str = add_0_str(day)
    m_str = add_0_str(m)
    r_str = add_00_str(today_run)

    g = exp_input.graph

    # code
    # solver type
    solver = ""
    solver_type = exp_input.solver_type
    if solver_type == 'central':
        solver = 'C'
    elif solver_type == 'distributed':
        solver = 'D'

    horizon = exp_input.horizon
    if horizon is None:
        H = ""
    else:
        H = 'H' + str(horizon)

    deadline = exp_input.deadline
    if deadline == horizon:
        Tau = ""
    else:
        Tau = 'T' + str(deadline)

    # number of searchers
    number_searchers = 'S' + str(exp_input.size_team)

    # graph used
    graph_number = ""
    if g["name"] == 'G60V.p':
        # MUSEUM
        graph_number = 1
    elif g["name"] == 'G100V_grid.p':
        # GRID 10x10
        graph_number = 2
    elif g["name"] == 'G70V_OFFICE.p':
        # OFFICE
        graph_number = 3

    graph_used = 'G' + str(graph_number)

    # false negative or not
    if exp_input.zeta is None or exp_input.zeta < 0.0001:
        meas_type = 'TN'
    else:
        meas_type = 'FN'

    # capture range
    if exp_input.capture_range == 0:
        cap_range = 'SV'
    else:
        cap_range = 'MV'

    name_code = solver + H + Tau + number_searchers + graph_used + meas_type + cap_range

    # assemble folder name
    name_folder = name_code + "_" + m_str + day_str + "_" + r_str
    whole_path = get_whole_path(name_folder, folder_parent)
    return name_folder, whole_path


def get_name_folder(today_run: int):
    """Give back the name of the folder
        which follows:
        today's date + _ + today's run
        if single digit, add zero in front"""

    # get todays date
    d = datetime.datetime.today()
    m = d.month
    day = d.day

    day_str = add_0_str(day)
    m_str = add_0_str(m)
    r_str = add_00_str(today_run)

    # assemble folder name
    name_folder = m_str + day_str + "_" + r_str
    whole_path = get_whole_path(name_folder, 'data')
    return name_folder, whole_path


def get_whole_path(name_folder: str, option='data'):

    # this file directory
    path1 = os.path.dirname(os.path.abspath(__file__))

    if 'core' in path1:
        # go up one level
        path_this_file = os.path.dirname(path1)
    else:
        path_this_file = path1

    # parent directory/data/date_run
    whole_path = path_this_file + "/" + option + "/" + name_folder

    return whole_path


def get_parent_folder(option='data'):
    # this file directory
    path1 = os.path.dirname(os.path.abspath(__file__))

    if 'core' in path1:
        # go up one level
        path_this_file = os.path.dirname(path1)
    else:
        path_this_file = path1

    my_path = path_this_file + "/" + option

    return my_path


def name_pickle_file(name_folder: str):

    file_path = get_whole_path(name_folder, 'data')

    filename = file_path + "/" + 'global_save.pickle'

    return filename


def create_my_folder(today_run=0):
    """Create directory if it doesnt exist, return the name of the directory (not the path)"""

    name_folder, whole_path = get_name_folder(today_run)

    print(whole_path)
    # create new folder to save figures
    if not os.path.exists(whole_path):
        os.mkdir(whole_path)
    else:
        print("Directory " + name_folder + " already exists")
    return name_folder


def create_my_new_folder(m_searcher, g, today_run=0, solver_type='central', zeta=None, capture_range=0,
                         horizon=None, day_start=None):

    name_folder, whole_path = get_name_code_folder(today_run, m_searcher, g, solver_type, zeta, capture_range,
                                                       horizon, day_start)

    # print(whole_path)
    # create new folder to save figures
    if not os.path.exists(whole_path):
        os.mkdir(whole_path)
    else:
        print("Directory " + name_folder + " already exists")
    return name_folder


def create_my_folder_mod(g, solver_input, searcher_input, day_start, today_run):

    # unpack things
    horizon = solver_input['horizon']
    solver_type = solver_input['solver_type']

    capture_range = searcher_input['capture_range']
    m = searcher_input['size_team']
    zeta = searcher_input['zeta']

    # create name with code
    name_folder, whole_path = get_name_code_folder(today_run, m, g, solver_type, zeta, capture_range, horizon,
                                                       day_start)

    # print(whole_path)
    # create new folder to save figures
    if not os.path.exists(whole_path):
        os.mkdir(whole_path)
    else:
        print("Directory " + name_folder + " already exists")
    return name_folder


# -------------------------------------
# extract data
# -------------------------------------
def get_pickle_file(path_folder: str, name_pickle='global_save.txt'):

    pickle_file = path_folder + "/" + name_pickle
    return pickle_file


def find_captor(searchers):

    for s_id in searchers.keys():
        s = searchers[s_id]
        if s.catcher is True:
            return s_id

    return None


# -------------------------------------
# sampling from distributions
# -------------------------------------
def get_sample_normal(n: int, mu, sigma):
    """Get n samples from the diosxtribution x ~N(mu, sigma)"""
    samples = np.random.normal(mu, sigma, n)

    return samples.tolist()


def get_random_seed():

    my_seed = datetime.datetime.now().timestamp()
    random.seed(my_seed)
    return


def get_np_random_seed():
    my_seed = datetime.datetime.now().timestamp()
    my_seed2 = random.seed(my_seed)
    np.random.seed(my_seed2)

    return


def sample_vertex(my_vertices: list, prob_move: list):
    """ sample 1 vertex with probability weight according to prob_move"""
    # uncomment for random seed
    get_random_seed()
    my_vertex = np.random.choice(my_vertices, None, p=prob_move)
    return my_vertex


# --------------------------------------
# graph related info
# --------------------------------------
def get_connectivity_matrix(g):
    """Return conenectivity matrix
    e_vu = 1 iff i-j are connected by an edge
    e_vu = 0 otherwise"""

    V, n = get_set_vertices(g)
    E = np.zeros((n, n))

    for v in V:
        v_idx = get_python_idx(v)

        for u in V:
            u_idx = get_python_idx(u)

            # get distance between vertices
            d = get_node_distance(g, v, u)

            # set element e_vu
            if d == 1:
                E[v_idx, u_idx] = 1

    return E.tolist()


def has_edge(E, v: int, u: int):

    v_idx = get_python_idx(v)
    u_idx = get_python_idx(u)

    e_vu = E[v_idx][u_idx]

    return e_vu


def get_length_short_paths(G):
    """Return array of lengths of shortest path [source][target]"""
    my_array = G["path_len"]
    return my_array


def get_node_distance(g, v1: int, v2: int):
    """v1, v2 are given as labels V = 1,...n"""

    spl = get_length_short_paths(g)
    v1_idx = get_python_idx(v1)
    v2_idx = get_python_idx(v2)

    distance = spl[v1_idx][v2_idx]

    return distance


def retrieve_graph(sim_data):
    # graph file and layout
    g = sim_data.g
    if 'grid' in g['name']:
        my_layout = g.layout("grid")
    else:
        my_layout = g.layout("kk")

    return g, my_layout


# -----------------------
# retrieve specific graphs
# -----------------------
def get_graph(graph_name):
    """Return graph, input can be string with the file name (reuse previous created graph),
     or a variable containing the graph itself"""
    # open file if its a string, or just pass the graph variable
    if isinstance(graph_name, str):
        graph_file = get_whole_path(graph_name, 'graphs')
        # get graph info: returns file object, mode read binary
        infile = open(graph_file, 'rb')
        G = pickle.load(infile)
        infile.close()
    else:
        G = graph_name
    return G


def get_graph_01():
    """Load Hollinger, 2009 middle graph from Fig 2 MUSEUM"""
    graph_name = 'G60V.p'
    # graph_file = get_whole_path(graph_name, 'graphs')
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_02():
    """Load 10x10 grid graph"""
    graph_name = 'G100V_grid.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_03():
    """Load 8x8 grid graph"""
    graph_name = 'G256V_grid.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_04():
    """Load 3x3 grid graph"""
    graph_name = 'G9V_grid.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_05():
    """Load G20_home graph"""
    graph_name = 'G20_home.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_06():
    """Load G20_home graph"""
    graph_name = 'G25_home.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


def get_graph_07():
    """Load Hollinger, 2009 middle graph from Fig 2 OFFICE"""
    graph_name = 'G70V_OFFICE.p'
    g = get_graph(graph_name)
    g["name"] = graph_name

    return g


# -----------------------
# convert data format
# -----------------------
def convert_list_array(A, opt: str):
    """Change from list to array or from array to list"""
    # change to list
    if opt == 'list':
        if not isinstance(A, np.ndarray):
            B = False
        else:
            B = A.tolist()
    # change to array
    elif opt == 'array':
        if not isinstance(A, list):
            B = False
        else:
            B = np.asarray(A)
    else:
        print("Wrong type option, array or list only")
        B = False

    return B


def path_to_xs(path: dict):
    """Convert from pi(s, t) = v
    to
    x_s(s, v, t) = 1"""

    x_searchers = {}

    for k in path.keys():

        # ignore first one (if it's temp_pi)
        if k == 'current_searcher':
            continue

        s, t = get_from_tuple_key(k)
        # get vertex searcher is currently in
        v = path.get((s, t))

        x_searchers[(s, v, t)] = 1

    return x_searchers


def get_all_from_xs(x_s):
    """Return list of (s, v, t, value) tuples from x_s"""

    my_list = []
    for k in x_s.keys():
        s, v, t = get_from_tuple_key(k)
        value = x_s.get(k)
        my_list.append((s, v, t, value))

    return my_list


def get_from_tuple_key(k):

    # x_searchers (s, v, t)
    if len(k) == 3:
        s = k[0]
        v = k[1]
        t = k[2]
        return s, v, t

    # path (pi or temp_pi) --> (s, t)
    elif len(k) == 2:
        s = k[0]
        t = k[1]
        return s, t
    else:
        print('Error, tuple is wrong dimension')
        return None


def create_2tuple_keys(list1, list2):
    seq = []

    for i in list1:
        for j in list2:
            seq.append((i, j))

    my_dict = create_dict(seq, [])

    return my_dict


def create_3tuple_keys(list1, list2, list3):
    """Create dictionary of the form: d(i,j,k) = int
    default int value: -1"""

    seq = []

    for i in list1:
        for j in list2:
            for k in list3:
                seq.append((i, j, k))

    my_dict = create_dict(seq, -1)

    return my_dict


def get_highest(my_dict: dict, my_keys):
    """Loop through keys and get highest value of dictionary
    if there are more than one highest value, pick the last"""

    high_value = 0
    idx = None

    for k in my_keys:
        my_value = my_dict[k]

        if high_value <= my_value:
            high_value = my_value
            idx = k
        else:
            continue

    return high_value, idx


def create_dict(my_keys, default_value):

    if isinstance(default_value, dict):
        print('Starting default value as {} will cause everything to have the same value')
        default_value = None

    my_dict = {}
    for k in my_keys:
        my_dict[k] = default_value
    return my_dict


def init_dict_variables(n_var: int):
    """create empty variables pertinent to the model"""
    # pack things
    my_vars = []
    for i in range(0, n_var):
        my_vars.append({})

    return my_vars


