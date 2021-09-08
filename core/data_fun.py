
import os
import pickle
import numpy as np
import scipy
from scipy.stats import sem as sem_sp, t as t_sp
from classes.belief import MyBelief
from core import extract_info as ext


def data_folder(folder_name='plot_data'):

    project_folder = ext.get_project_path()
    folder_data_path = project_folder + '/' + folder_name

    exp_configs = os.listdir(folder_data_path)

    subfolders_name = dict()
    subfolders_path = dict()

    # folder: CH6-20S1G1TNSV
    for folder_name in exp_configs:
        # print(folder_name)

        path_folder = folder_data_path + "/" + folder_name

        # simulation instances
        my_instances = os.listdir(path_folder)
        subfolders_name[folder_name] = my_instances

        subfolders_path[folder_name] = []
        for instance in my_instances:
            instance_path = path_folder + "/" + instance
            subfolders_path[folder_name].append(instance_path)

    return subfolders_path


def specific_from_data_folder(folder_names, folder_name='data_review'):

    project_folder = ext.get_project_path()
    folder_data_path = project_folder + '/' + folder_name

    exp_configs = os.listdir(folder_data_path)

    subfolders_name = dict()
    subfolders_path = dict()

    # folder: CH6-20S1G1TNSV
    for folder_name in exp_configs:
        # print(folder_name)

        if folder_name in folder_names:

            path_folder = folder_data_path + "/" + folder_name

            # simulation instances
            my_instances = os.listdir(path_folder)
            subfolders_name[folder_name] = my_instances

            subfolders_path[folder_name] = []
            for instance in my_instances:
                instance_path = path_folder + "/" + instance
                subfolders_path[folder_name].append(instance_path)

    return subfolders_path, subfolders_name


def load_data(folder_path, file_name='/global_save.txt'):
    """unpickle the data file"""

    if '/' not in file_name:
        file_name = '/' + file_name

    file_path = folder_path + file_name

    try:
        data = pickle.load(open(file_path, "rb"))

    except:
        print('Make sure your parameters are right!')
        data = None

    return data


def get_classes(data):

    belief = data["belief"]
    target = data["target"]
    searchers = data["searchers"]
    solver_data = data["solver_data"]

    try:
        exp_inputs = data["inputs"]
    except:
        exp_inputs = None

    return belief, target, searchers, solver_data, exp_inputs


def get_from_inputs(exp_inputs):
    """retrieve: h, gamma, n, v0, b(0), inter"""

    h = exp_inputs.horizon

    gamma = exp_inputs.gamma

    g = exp_inputs.graph

    n = len(g.vs)

    inter = 1

    return g, h, gamma, n, inter


def get_from_target(target):

    M = target.motion_matrix

    return M


def get_from_belief(belief):
    b0 = belief.start_belief

    return b0


def generate_txt(data_list, fname, my_path):

    file_path = my_path + '/' + fname + '.txt'

    f = open(file_path, 'w')

    for el in data_list:
        f.write(el + '\n')

    f.close()


def get_from_searchers(searchers):

    v0 = []
    C = dict()

    for i in searchers.keys():
        s = searchers[i]
        v0.append(s.start)
        C[i] = s.capture_matrices

    m = len(searchers.keys())
    return m, v0, C


def get_from_solver_data(solver_data, m):

    t = 0

    # get cost function
    cost_t = solver_data.obj_value[t]
    # get solver_time
    sol_t = solver_data.solve_time[t]

    if solver_data.solver_type == 'distributed':
        obj_value = cost_t[m]
        sol_time = sol_t['total']
    else:
        # get cost function
        obj_value = cost_t
        # get solver_time
        sol_time = sol_t

    # get computed path
    computed_path = solver_data.x_s[t]

    return computed_path, obj_value, sol_time


def organize_data_make_files(belief, target, searchers, solver_data, exp_inputs, sub_path):

    # recover relevant data
    # recover relevant data
    g, h, gamma, n, inter = get_from_inputs(exp_inputs)
    b0 = get_from_belief(belief)
    m, v0, C = get_from_searchers(searchers)
    M = get_from_target(target)
    computed_path, obj_value, sol_time = get_from_solver_data(solver_data, m)

    # reformat if necessary
    b_0 = list_to_str(b0)
    v_0 = list_to_str(v0)
    M_as_line = format_motion_matrix(M, n)
    C_line_list= format_C_matrices(C, n)
    adj_line_list = format_connectivity(g, n)
    try:
        path_line_list = format_path(computed_path)
        generate_txt(path_line_list, 'path', sub_path)
    except:
        return

    # generate txt files
    generate_txt([str(h), str(gamma), str(n), v_0, b_0, str(inter)], 'start', sub_path)
    generate_txt([M_as_line], 'M', sub_path)
    generate_txt(C_line_list, 'C', sub_path)
    generate_txt(adj_line_list, 'adj', sub_path)
    generate_txt([str(sol_time), str(obj_value)], 'solver_data', sub_path)


def list_to_str(my_list):

    my_str = ''
    for el in my_list:
        my_str = my_str + str(el) + space()

    return my_str


def format_path(computed_path):

    path_list = []

    for s in computed_path.keys():

        s_pos = computed_path[s]

        s_pi = str(s) + space()

        for el in s_pos:
            v = el[0]
            t = el[1]

            if s_pos.get((v, t)) == 1:
                s_pi += str(v) + space()

        path_list.append(s_pi)

    return path_list


def format_motion_matrix(M, n):

    # 1st row: [1 0(1xn)]
    first_row = '1 '
    for i in range(0, n):
        first_row += str(0.0) + space()

    my_str = ''
    for row in M:
        # put 0.0 in beginning of each row
        my_str += str(0.0) + space()
        # values from M row
        for el in row:
            my_str += str(el) + space()

    big_M = first_row + my_str

    return big_M


def format_C_matrices(C, n):
    """line: searcher_id, vertex_id, capture_matrix"""

    # list of strings -- each string is a a line
    C_for_file = []

    for s in C.keys():
        for v in list(range(1, n+1)):
            Csv = C[s][v]
            C_row = str(s) + space() + str(v) + space()
            for array_row in Csv:
                for el in array_row:
                    C_row += str(el) + space()

            C_for_file.append(C_row)

    return C_for_file


def format_connectivity(g, n):
    """adjacency list describing the graph. 1 vertex for each line. Example:
    1 2 4
    """

    # list of strings
    con_list = []

    # loop through vertices
    for v_idx in range(0, n):

        # connections of v as string
        con_v = str(v_idx + 1) + space()

        v_nei = g.vs[v_idx]["neighbors"]

        for u_idx in v_nei:
            con_v += str(u_idx + 1) + space()

        con_list.append(con_v)

    return con_list


def space():
    return ' '


def get_single_file(folder='data', instance_n=11):

    project_folder = ext.get_project_path()
    folder_data_path = project_folder + '/' + folder

    all_subs = os.listdir(folder_data_path)

    for folder in all_subs:

        this_ins = folder.split('_')[-1]

        if instance_n == int(this_ins):
            print('%s - ' % instance_n, sep=' ', end='', flush=True)

            my_path = folder_data_path + '/' + folder
            inf_ = get_from_path(my_path, 'inf')

            return inf_


def get_from_path(my_path: str, op='cpp'):
    # get data
    data = load_data(my_path)

    # classes
    belief, target, searchers, solver_data, exp_inputs = get_classes(data)

    # # t, s
    # i, j = 3, 5
    #
    # sum_t_1 = sum(belief.stored[i-1])
    # sum_1 = sum(belief.stored[i])
    # path_t_1 = searchers[j].path_planned[i-1]
    # path_t = searchers[j].path_planned[i]
    #
    # print('belief sum at t = %d %.4f' % (i-1, sum_t_1))
    # print('belief sum at t = %d (inf) %.4f' % (i, sum_1))
    # print('path planned for s = ' + str(j) + ' at t-1 inf' + str(path_t_1))
    # print('path planned for s = ' + str(j) + ' at t inf' + str(path_t))

    if op == 'cpp':
        organize_data_make_files(belief, target, searchers, solver_data, exp_inputs, my_path)

    elif op == 'inf':

        if print_inf(searchers, solver_data, exp_inputs):
            return int(1)
        else:
            return int(0)

    else:
        print('Select a valid option.')

    return


def print_inf(searchers, solver_data, exp_inputs):

    is_inf = False

    aux_ = exp_inputs.name_folder.split('/')
    ins_name = aux_[-1].split('_')[-1]

    max_t = len(solver_data.obj_value.keys())

    for t in solver_data.obj_value.keys():

        # get obj value for all searchers
        cost_fun_t = []
        for s in searchers.keys():
            cost_fun_t.append(solver_data.obj_value[t][s])

        # if there is an infeasible instance
        if -1 in cost_fun_t:

            if not is_inf:
                print('\n\n-------- Instance %s --------' % ins_name)

            print('Problem was infeasible at time-step %d' % t)
            is_inf = True

            # get positions to make sure no one is tele-transporting
            pos_t = [searchers[s].path_taken[t] for s in searchers.keys()]
            p_str = str(pos_t)
            print('Searchers position at t = %d: %s X-X-X' % (t, p_str))

            if t < max_t:
                # check next position (still)
                pos_t_next = [searchers[s].path_taken[t + 1] for s in searchers.keys()]
                p_str_next = str(pos_t_next)
                print('Searchers position at t = %d: %s ' % (t + 1, p_str_next))

                if t < max_t - 1:
                    pos_t_after = [searchers[s].path_taken[t + 2] for s in searchers.keys()]
                    p_str_after = str(pos_t_after)
                    print('Searchers position at t = %d: %s ' % (t + 2, p_str_after))
            print('\n')

    if is_inf:
        return True

    return False


def check_sol():

    list_folders = ['DH2-8S3G1TNSV MILP', 'DH2-8S3G1TNSV CPP', 'DH2-8S3G3TNSV MILP', 'DH2-8S3G3TNSV CPP',
                    'DH2-8S3G2FNMV MILP', 'DH2-8S3G2FNMV CPP']

    folder_name = 'data_review'

    # keys of dicts to store data
    list_h = [2, 4, 6, 8]
    k_milp = 'milp'
    k_cpp = 'cpp'
    sol_time_museum, sol_time_gridfn, sol_time_office = {}, {}, {}

    subfolders_path, subfolders_name = specific_from_data_folder(list_folders, folder_name)

    for folder in list_folders:

        sol_time_milp = ext.create_dict(list_h, None)
        sol_time_cpp = ext.create_dict(list_h, None)

        obj_fun_milp, obj_fun_cpp = [], []

        if 'MILP' not in folder:
            continue

        i = 0

        milp_paths = subfolders_path[folder]
        milp_names = subfolders_name[folder]

        j = list_folders.index(folder) + 1

        cpp_paths = subfolders_path[list_folders[j]]
        cpp_names = subfolders_name[list_folders[j]]

        print('\nStarting %s' % folder)
        # retrieve data from the txt files: path -- time_sol -- obj _fun
        for sub_name in subfolders_name[folder]:

            # print instance
            horizon = sub_name.split('S')[0][1:]
            ins_name = sub_name.split('_')[-1]

            h = int(horizon[-1])

            if sol_time_milp[h] is None:
                sol_time_milp[h] = []
                sol_time_cpp[h] = []

            my_text = '%s %s.%d -- ' % (horizon, ins_name, i)
            print(my_text, sep=' ', end='', flush=True)
            i += 1

            # find index in folder
            idx = subfolders_name[folder].index(sub_name)

            # check if it's the same
            if not milp_names[idx] == cpp_names[idx]:
                print('Error on %s' % milp_names[idx])
                break

            milp_ins_path = milp_paths[idx]
            cpp_ins_path = cpp_paths[idx]

            # get milp data
            milp_plan = get_from_txt(milp_ins_path, 'path')
            milp_solver = get_from_txt(milp_ins_path, 'solver_data')

            cpp_plan = get_from_txt(cpp_ins_path, 'paths')
            cpp_solver = get_from_txt(cpp_ins_path, 'solver_data')

            # check consistency between paths

            dc = 4
            # append obj fun and time sol
            obj_fun_cpp.append(round(cpp_solver[2][0], dc))
            obj_fun_milp.append(round(milp_solver[2][0], dc))

            sol_time_cpp[h].append(round(cpp_solver[1][0], dc))
            sol_time_milp[h].append(round(milp_solver[1][0], dc))

            # time_difference(sol_time_milp[h], sol_time_cpp[h])
            # check_timeout(sol_time_milp[h])

        print("\n")

        check_obj(obj_fun_milp, obj_fun_cpp)

        # -----------------------
        # save data
        # -----------------------

        if 'G1' in folder:
            sol_time_museum = dict()
            sol_time_museum[k_milp] = sol_time_milp
            sol_time_museum[k_cpp] = sol_time_cpp

        elif 'G2' in folder:
            sol_time_gridfn = dict()
            sol_time_gridfn[k_milp] = sol_time_milp
            sol_time_gridfn[k_cpp] = sol_time_cpp

        elif 'G3' in folder:
            sol_time_office = dict()
            sol_time_office[k_milp] = sol_time_milp
            sol_time_office[k_cpp] = sol_time_cpp
        else:
            print('Error! Check which graph this is.')
            exit()

    return sol_time_museum, sol_time_gridfn, sol_time_office


def compute_stats(sol_time_museum, sol_time_gridfn, sol_time_office):

    milp = 'milp'
    cpp = 'cpp'
    list_h = [2, 4, 6, 8]

    list_dicts = [sol_time_museum, sol_time_gridfn, sol_time_office]

    avg_time_museum, avg_time_gridfn, avg_time_office = {}, {}, {}
    conf_time_museum, conf_time_gridfn, conf_time_office = {}, {}, {}

    i = 1
    for dic in list_dicts:

        sol_time_avg = {milp: [], cpp: []}
        sol_time_confidence = {milp: [], cpp: []}

        list_milp, list_cpp = [], []
        for h in list_h:
            list_milp.append(dic[milp][h])
            list_cpp.append(dic[cpp][h])

        # get average
        my_avg, my_conf = get_confidence(list_milp)

        sol_time_avg[milp] = my_avg
        sol_time_confidence[milp] = my_conf

        my_avg, my_conf = get_confidence(list_cpp)

        sol_time_avg[cpp] = my_avg
        sol_time_confidence[cpp] = my_conf

        my_env = ''
        if i == 1:
            avg_time_museum = sol_time_avg
            my_env = 'MUSEUM'

        elif i == 2:
            avg_time_gridfn = sol_time_avg
            my_env = 'GRID-FN'

        elif i == 3:
            avg_time_office = sol_time_avg
            my_env = 'OFFICE'

        else:
            exit()

        list1 = [round(num, 3) for num in sol_time_avg[milp]]
        list3 = [round(num, 4) for num in sol_time_confidence[milp]]

        list2 = [round(num, 3) for num in sol_time_avg[cpp]]
        list4 = [round(num, 4) for num in sol_time_confidence[cpp]]

        r_milp_cpp = [round(list2[i]/list1[i], 2) for i in range(len(list1))]

        print('\n%s: h = [2, 4, 6, 8]' % my_env)
        print('milp avg: ' + str(list1))
        print('cpp avg: ' + str(list2))

        print('===\nratio cpp/milp ' + str(r_milp_cpp) + '\n===')


        print('\nmilp conf: ' + str(list3))
        print('cpp conf: ' + str(list4))

        i += 1

    return avg_time_museum, avg_time_gridfn, avg_time_office


def print_sloppy():
    sol_time_museum, sol_time_gridfn, sol_time_office = check_sol()
    avg_time_museum, avg_time_gridfn, avg_time_office = compute_stats(sol_time_museum, sol_time_gridfn, sol_time_office)


def check_timeout(sol_time_milp):

    counter = 0
    for el in sol_time_milp:
        if el > 120:
            counter += 1

    print('MILP over timeout %d' % counter)


def check_obj(obj_1, obj_2):

    my_list = list(range(len(obj_1)))

    for idx in my_list:

        fun_1 = obj_1[idx]
        fun_2 = obj_2[idx]

        if fun_1 - fun_2 < 0.01:
            continue

        print('Obj Function is different in idx %d: MILP %.4f CPP %.4f' % (idx, fun_1, fun_2))


def time_difference(sol_1, sol_2):
    my_list = list(range(len(sol_1)))

    ratio = []

    for idx in my_list:

        fun_1 = sol_1[idx]
        fun_2 = sol_2[idx]

        ratio.append(round(fun_2/fun_1, 2))

    avg_ratio = np.mean(ratio)
    print('Mean ratio time sol CPP/MILP = %.2f' % avg_ratio)


def get_from_txt(my_path, name_txt):
    """Return as dictionary of lists
    data[line #] = [int or floats] """

    f_name = my_path + '/' + name_txt + '.txt'
    f = open(f_name, 'r')

    data = {}

    i = 0
    for line in f.readlines():
        i += 1
        data[i] = []

        str_line = line.split(' ')
        for el in str_line:
            try:
                if 'path' in name_txt:
                    data[i].append(int(el))
                else:
                    data[i].append(float(el))
            except:
                continue

    f.close()
    return data


def my_script():

    folder_name = 'data_review'

    list_not = []  # ['CH10S1-5G1TNSV', 'DH10S1-5G1TNSV', 'DH10S1-5G2FNMV', 'plan_only']

    list_yes = ['DH10S1-5G2FNMV']

    subfolders_path = data_folder(folder_name)

    for folder in subfolders_path.keys():

        if folder in list_not:
            continue

        if folder not in list_yes:
            continue

        i = 0
        print('Starting %s' % folder)
        for sub in subfolders_path[folder]:

            # print instance
            horizon = sub.split('/')[-1].split('S')[0][1:]
            ins_name = sub.split('_')[-1]
            my_text = '%s %s-%d -.- ' % (horizon, ins_name, i)
            print(my_text, sep=' ', end='', flush=True)

            # instance path
            my_path = sub

            # get data
            data = load_data(my_path)

            if data is None:
                print('X - ', sep=' ', end='', flush=True)
                continue

            # classes
            belief, target, searchers, solver_data, exp_inputs = get_classes(data)

            organize_data_make_files(belief, target, searchers, solver_data, exp_inputs, my_path)

            i += 1

        print('Processed %d instances in %s' % (i, folder))


def make_searcher_info(m, n, s_0, C_lines):

    zeta = 0.3
    my_type = 'int32'

    if zeta is not None:
        my_type = 'float64'

    S = list(range(1, m+1))

    V = list(range(1, n+1))

    s_info = {}

    for s in S:
        s_C = {}
        for v in V:
            s_C[v] = None

            for line in C_lines.keys():
                s_id = int(C_lines[line][0])
                v_id = int(C_lines[line][1])

                if s_id == s and v_id == v:
                    C_as_list = C_lines[line][2:]

                    C = get_C(C_as_list, n)
                    C_arr = np.array(C, my_type)
                    s_C[v] = C_arr
                    break

        idx = s - 1
        s_info.update({s: {'start': s_0[idx], 'c_matrix': s_C, 'zeta': zeta}})

    return s_info


def get_C(C_as_list, n):

    nu = n + 1

    i_s = 0
    C = []
    for row in range(nu):

        C.append([])

        i_f = i_s + n + 1
        my_r = C_as_list[i_s:i_f]

        C[row] = my_r
        i_s = i_f

        assert len(C[row]) == nu
    assert len(C) == nu

    return C


def get_M(M_line, n):

    # take out first line [1 0_1xn]
    bigM = M_line[1][n+1:]

    assert len(M_line[1]) == (n+1)*(n+1)
    assert len(bigM) == n*(n+1)

    M = []
    i_s = 0
    for row in range(n):

        M.append([])
        i_f = i_s + n + 1
        my_r = bigM[i_s:i_f]

        M[row] = my_r[1:]
        i_s = i_f

        assert len(M[row]) == n
    assert len(M) == n

    return M


def get_start_specs(start_data):
    """Get number fo searchers (m), gamma, start position of searchers (s_0), initial belief vector"""

    h = int(start_data[1][0])

    gamma = start_data[2][0]

    n = int(start_data[3][0])

    s_0 = [int(start_data[4][j]) for j in range(len(start_data[4]))]

    b_0 = start_data[5]

    return h, n, gamma, s_0, b_0


def check_milp():

    folder_milp = 'data_review/DH10S1-5G2FNMV'

    list_ins = ['DH10S1G2FNMV_0513_011', 'DH10S3G2FNMV_0513_211', 'DH10S5G2FNMV_0513_460']

    for instance in list_ins:
        my_path = ext.get_project_path() + '/' + folder_milp + '/' + instance

        print('\n===================================\nTesting %s \n===================================' % instance)

        # collect data
        start_data = get_from_txt(my_path, 'start')
        M_line = get_from_txt(my_path, 'M')
        C_lines = get_from_txt(my_path, 'C')

        solver_data_milp = get_from_txt(my_path, 'solver_data')

        # relevant parameters/info
        h, n, gamma, s_0, b_0 = get_start_specs(start_data)
        s_plan_milp = get_from_txt(my_path, 'path')

        m = len(s_plan_milp)
        M = get_M(M_line, n)
        s_info = make_searcher_info(m, n, s_0, C_lines)
        S = range(1, m + 1)

        obj_fun_milp = solver_data_milp[2][0]

        # compute collected reward according to plan

        # init belief
        belief_milp = MyBelief(b_0)

        rwd_milp = 0.0

        for t in range(0, h + 1):

            # evolve searchers positions
            pi_next_t_milp = {}

            v_milp = []

            for s_id in S:

                next_v_milp = s_plan_milp[s_id][t + 1]

                v_milp.append(next_v_milp)

                pi_next_t_milp[s_id] = next_v_milp

            if t > 0:
                belief_milp.update(s_info, pi_next_t_milp, M, n)

            b_c_milp = belief_milp.stored[t][0]

            print('\n---- t = %d' % t)

            print('Vertices MILP: ' + str(v_milp))

            print('Belief MILP : %.4f' % b_c_milp)

            rwd_t_milp = (gamma ** t) * b_c_milp

            print('Reward MILP: %.4f' % rwd_t_milp)

            rwd_milp += rwd_t_milp

        print('\n:.:Total Reward MILP: %.4f' % rwd_milp)
        print(':.:Cost Fun from MILP: %.4f' % obj_fun_milp)


def find_inf():
    my_folder = 'data_review/DH10S1-5G2FNMV'

    my_list = list(range(1, 501)) # [11, 111, 211, 311, 411]  #

    inf_counter = 0

    for el in my_list:
        inf_counter += get_single_file(my_folder, el)

    print('Inf instances: %d ' % inf_counter)


def check_milp_vs_cpp_sum():

    folder_milp = 'data_review/DH2-8S3G3TNSV MILP'
    folder_cpp = 'data_review/DH2-8S3G3TNSV CPP'

    list_ins = ['DH4S3G1TNSV_0512_023', 'DH4S3G1TNSV_0512_078', 'DH4S3G1TNSV_0512_103']

    for instance in list_ins:
        my_path = ext.get_project_path() + '/' + folder_milp + '/' + instance

        path_cpp = ext.get_project_path() + '/' + folder_cpp + '/' + instance

        print('\n===================================\nTesting %s \n===================================' % instance)

        # collect data
        start_data = get_from_txt(my_path, 'start')
        M_line = get_from_txt(my_path, 'M')
        C_lines = get_from_txt(my_path, 'C')

        solver_data_milp = get_from_txt(my_path, 'solver_data')
        solver_data_cpp = get_from_txt(path_cpp, 'solver_data')

        # relevant parameters/info
        h, n, gamma, s_0, b_0 = get_start_specs(start_data)
        s_plan_milp = get_from_txt(my_path, 'path')
        s_plan_cpp = get_from_txt(path_cpp, 'paths')

        m = len(s_plan_milp)
        M = get_M(M_line, n)
        s_info = make_searcher_info(m, n, s_0, C_lines)
        S = range(1, m + 1)

        obj_fun_milp = solver_data_milp[2][0]
        obj_fun_cpp = solver_data_cpp[2][0]

        # compute collected reward according to plan

        # init belief
        belief_milp = MyBelief(b_0)
        belief_cpp = MyBelief(b_0)

        rwd_milp = 0.0
        rwd_cpp = 0.0

        for t in range(0, h + 1):

            # evolve searchers positions
            pi_next_t_milp = {}
            pi_next_t_cpp = {}

            v_milp, v_cpp = [], []

            for s_id in S:

                next_v_milp = s_plan_milp[s_id][t + 1]
                next_v_cpp = s_plan_cpp[s_id][t + 1]

                v_milp.append(next_v_milp)
                v_cpp.append(next_v_cpp)

                pi_next_t_milp[s_id] = next_v_milp
                pi_next_t_cpp[s_id] = next_v_cpp

            if t > 0:
                belief_milp.update(s_info, pi_next_t_milp, M, n)
                belief_cpp.update(s_info, pi_next_t_cpp, M, n)

            b_c_milp = belief_milp.stored[t][0]
            b_c_cpp = belief_cpp.stored[t][0]

            print('\n---- t = %d' % t)

            print('Vertices MILP: ' + str(v_milp))
            print('Vertices CPP: ' + str(v_cpp))

            print('Belief MILP : %.4f' % b_c_milp)
            print('Belief CPP : %.4f' % b_c_cpp)

            rwd_t_milp = (gamma ** t) * b_c_milp
            rwd_t_cpp = (gamma ** t) * b_c_cpp

            print('Reward MILP: %.4f' % rwd_t_milp)
            print('Reward CPP: %.4f' % rwd_t_cpp)

            rwd_milp += rwd_t_milp
            rwd_cpp += rwd_t_cpp

        print('\n:.:Total Reward MILP: %.4f' % rwd_milp)
        print(':.:Cost Fun from MILP: %.4f' % obj_fun_milp)

        print('\n:.:Total Reward CPP: %.4f' % rwd_cpp)
        print(':.:Cost Fun from CPP:  %.4f' % obj_fun_cpp)


def create_txt(name_folder):

    my_path = ext.this_folder(name_folder, 'data_plan')

    # get data
    data = load_data(my_path)

    if data is None:
        print('X - ', sep=' ', end='', flush=True)

    # classes
    belief, target, searchers, solver_data, exp_inputs = get_classes(data)

    organize_data_make_files(belief, target, searchers, solver_data, exp_inputs, my_path)


def get_confidence(some_list: list):
    """Get confidence interval of 95%"""

    confidence = 0.68

    m = []
    y_err = []
    max_j = len(some_list)

    for i in range(0, max_j):
        n = len(some_list[i])
        m_i = scipy.mean(some_list[i][:])
        std_err = sem_sp(some_list[i][:])
        h = std_err * t_sp.ppf((1 + confidence) / 2, n - 1)

        m.append(m_i)
        y_err.append(h)

    return m, y_err


if __name__ == "__main__":

    # my_script()
    # find_inf()
    # check_sol()
    print_sloppy()
    # check_milp_vs_cpp_sum()
    # check_milp()
    # get_single_file('data_review/DH10S1-5G2FNMV', 11)




"""start.txt : 
1st line, planning horizon; 2nd line, the discount factor; 3rd line, the number of graph vertices; 
4th line, the list of robots’ starting vertices; 5th line, the initial belief vector (including b0); 
6th line, the number of “outer iterations” of the algorithm (typically 1).
Example:
10  h
0.99 gamma
2 n
1 2 1 v0
0.0 0.2 0.8 …… b(0)
1 (inter)

M.txt: argets’ motion model matrix, including the 1st row / column with zeros as a flattened array. 
[1 0; 0 M] → bigM as 1 row

C.txt: file containing the capture matrices as flattened arrays. Syntax: searcher_id, vertex_id, capture_matrix

adj.txt: file containing the adjacency list describing the graph. 1 vertex for each line.


paths.txt: file containing, for each line, the path computed for each robot (from step 0/starting vertex)

solver_data.txt: file containing the solver results. 1st line time in seconds, 2nd line objective function value. 

"""


