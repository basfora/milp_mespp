import sys
import os
import gflags
import matplotlib.pyplot as plt
import pickle
import numpy as np
from core import extract_info as ext

this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_path)

plt.rcParams.update({'font.size': 16})

# launch the script from the experiments folder.

# log_folder is expressed w.r.t. the experiments folder
gflags.DEFINE_string('log_path', '../plot_data/', 'path_to_folder_of_results')
# How many runs we have for each setting
gflags.DEFINE_integer('n_runs', 100, 'number of runs for each experiment')
#
#------------------------------------------------------------------------------------------
# this flag specifies what plot you want to do (e.g. plots for centralized experiments will be different from plots
# for distributed experiments. I am leaving in this script just the centralized for clarity.
gflags.DEFINE_string('solver_type', 'C', 'C = centralized')
# C = centralized, D = distributed, X = CxD

# The planning horizon used by the experiments
gflags.DEFINE_list('h_list', [10], 'horizons tested')

# 6, 8, 10, 12, 14, 16, 18, 20

# The deadline used in the experiments
gflags.DEFINE_integer('tau', 50, 'deadline')

# The team sizes used in the experiments
gflags.DEFINE_list('m_list', [1, 2, 3, 4, 5], 'm')


gflags.DEFINE_list('my_list', [], 'variable values')

gflags.DEFINE_string('key_l', 'm', 'experiment varies m, h, tau')

# Which type of plot I want
gflags.DEFINE_integer('type_plot', 1, 'type of plot I want')
# 1: Solution time and MIP gaps MUSEUM, GRID-NOFN, GRID-FN (old style) - centralized, varying m
# 2: Solution time and MIP gaps MUSEUM, GRID-NOFN, GRID-FN (old style) - centralized, varying h
#
# 3: Solution time and MIP gaps MUSEUM, GRID-FN (old style) - distributed, varying m
#
# 4: Objective Function and average mission time (distributed h5 x distributed h10, same graph) - deadline 50, varying m
#
# 5: Objective Function and average mission time (centralized x distributed, same graph) - deadline 10, varying m
# #


# --------------------------------------------------------------------------------------------------------------------
# deciding what to get
#
# assemble parents folder name
def my_exp_encoding():
    # assemble basename according to gflags

    # unpack

    scenarios, solver_type, key_l, my_list = set_flags()

    my_plot = gflags.FLAGS.type_plot

    m = gflags.FLAGS.m_list[0]
    h = gflags.FLAGS.h_list[0]

    # put together name of folder (start of name)
    if key_l == 'S':
        # varying m
        a1 = str(h)
        a2 = str(my_list[0]) + '-' + str(my_list[-1])

        st_base = solver_type + 'H' + a1 + 'S'
        end_base = ""

        if my_plot == 4:
            # add the tau
            a1 = str(h) + 'T' + str(gflags.FLAGS.deadline)

    elif key_l == 'H':
        # varying h
        a1 = str(my_list[0]) + '-' + str(my_list[-1])
        a2 = str(m)

        st_base = solver_type + 'H'
        end_base = 'S' + a2

    else:
        a1, a2, st_base, end_base = None, None, None, None
        print('check your flags')
        exit()

    # assemble start
    aux_name = solver_type + 'H' + a1 + 'S' + a2

    # append end of name (scenario)
    settings = []
    base_name = {'st': [], 'end': []}

    for el in scenarios:
        settings.append(aux_name + el)

        base_name['st'].append(st_base)
        base_name['end'].append(end_base + el)

    print(settings)
    return settings, base_name


def set_flags():
    scenarios = get_my_scenarios()
    solver_type = get_solver_type()
    key_l, my_list = get_var_interest()

    return scenarios, solver_type, key_l, my_list


# based on type of plot, get my scenarios accordingly
def get_my_scenarios():

    my_plot = gflags.FLAGS.type_plot

    settings = []

    if my_plot <= 2:
        settings = ['G1TNSV', 'G2TNMV', 'G2FNMV']
    elif my_plot <= 5:
        settings = ['G1TNSV', 'G2FNMV']
    else:
        print('No settings found')
        exit()

    return settings


def get_solver_type():

    my_plot = gflags.FLAGS.type_plot
    solver_type = ""

    if my_plot <= 2:
        solver_type = 'C'
    elif my_plot <= 4:
        solver_type = 'D'
    elif my_plot == 5:
        solver_type = 'CD'
    else:
        print('No settings found')
        exit()

    gflags.FLAGS.solver_type = solver_type
    return solver_type


def get_var_interest():
    my_plot = gflags.FLAGS.type_plot

    if my_plot == 2:
        key_l = 'H'
        my_list = gflags.FLAGS.h_list
    else:
        key_l = 'S'
        my_list = gflags.FLAGS.m_list

    gflags.FLAGS.key_l = key_l
    gflags.FLAGS.my_list = my_list

    return key_l, my_list


def get_label():

    key_l = gflags.FLAGS.key_l

    if key_l == 'H':

        y_label = '$h$'
    else:
        y_label = '$m$'

    my_list = gflags.FLAGS.my_list

    return my_list, y_label


def get_folder_path(e):
    folder_interest = gflags.FLAGS.log_path + e

    # sub folders of my folder of interest
    subfolders = os.listdir(folder_interest)

    return folder_interest, subfolders


def get_number_file(key_l, my_el, my_list, i):

    n_id = my_list.index(my_el)

    # fix the numbering if needed
    if key_l == 'H':
        if my_el == 10:
            n_id = 0
        elif my_el == 12:
            n_id = 2

    n_file = n_id * gflags.FLAGS.n_runs + i

    return n_file


# --------------------------------------------------------------------------------------------------------------------
# basic plot functions
# this is a generic function to draw a box plot.
def draw_box_plot(data, ax, edge_color, fill_color):

    # actual plotting
    bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    # plt.setp(bp['medians'], linewidth=2)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def draw_errorbar(xvalues, yvalues, std_dev, ax, l_style='bo-', lgd=None):
    ax.errorbar(xvalues, yvalues, yerr=std_dev, fmt=l_style, barsabove=False, label=lgd)
    ax.grid(b=True, which='both')


# set plot colors etc
def plot_gaps(gaps, my_list, y_label, filename):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # box plot with edge = black, fill = red
    draw_box_plot(gaps, ax, 'black', 'red')

    # thicks
    ax.set_xticklabels(map(lambda x: str(x), my_list))
    # ax.set_yticks(range(0, 30, 5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.set_ylim(-0.5, 32)

    # labels
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('MIP Gap [ % ]', fontsize=15)

    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


def plot_times(times, my_list, y_label, filename):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # box plot with edge = black, fill = blue
    draw_box_plot(times, ax, 'black', 'blue')

    # thicks
    # ax.set_yticks(range(0, 2000, 200))
    ax.set_xticklabels(map(lambda x: str(x), my_list))

    # plot labels
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('Solving Time [s]', fontsize=15)

    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


def plot_obj(avg, std_devs, my_list, y_label, filename):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    draw_errorbar(my_list, avg, std_devs, ax)

    # thicks
    ax.set_xticks(my_list)
    ax.set_ylim(0, max(avg) + max(std_devs)+0.1)

    # plot labels
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('Objective Function [avg]', fontsize=15)

    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


# --------------------------------------------------------------------------------------------------------------------
# parent plotting


def plot_for_me():
    """choose code according to which plot you want
    # 1: Solution time and MIP gaps MUSEUM, GRID-NOFN, GRID-FN (old style) - centralized, varying m
    # 2: Solution time and MIP gaps MUSEUM, GRID-NOFN, GRID-FN (old style) - centralized, varying h
    #
    # 3: Solution time and MIP gaps MUSEUM, GRID-FN (old style) - distributed, varying m
    #
    # 4: Objective Function and average mission time (distributed h5 x distributed h10, same graph) - deadline 50, varying m
    #
    # 5: Objective Function and average mission time (centralized x distributed, same graph) - deadline 10, varying m
    """

    my_plot = gflags.FLAGS.type_plot
    if my_plot <= 3:
        plot_results()
    elif my_plot <= 5:
        print('write')
    else:
        print('No such type registered')
        exit()


def plot_obj_multi(avg, std_devs, my_list, y_label, filename):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    l_style = {'C': 'co-', 'D': 'bo-'}

    for key in avg.keys():
        draw_errorbar(my_list, avg[key], std_devs[key], ax, l_style[key], key)


    ax.set_xticks(my_list)
    # ax.set_ylim(0, max(avg) + max(std_devs) + 0.1)

    # plot labels
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('Reward Function [avg]', fontsize=15)

    plt.legend(loc='lower right')
    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


def plot_obj_multi2(avg, my_list, y_label, filename):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    l_style = {'G1TNSV': 'co-', 'G2FNMV': 'bo-'}
    lgd = {'G1TNSV': 'MUSEUM', 'G2FNMV': 'GRID-FN'}

    for key in avg.keys():
        draw_errorbar(my_list, avg[key], None, ax, l_style[key], lgd[key])


    ax.set_xticks(my_list)
    # ax.set_ylim(0, max(avg) + max(std_devs) + 0.1)

    # plot labels
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('Reward Difference [avg]', fontsize=15)

    plt.legend(loc='lower right')
    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


def plot_capture_times(average_times, std_devs, my_list, y_label, filename):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    for h in average_times.keys():
        avg_t = average_times[h]
        std_dev = std_devs[h]
        if h == 6:
            ax.errorbar(my_list, avg_t, yerr=list(map(lambda x: 1.96 * x / gflags.FLAGS.n_runs, std_dev)),
                        fmt='ro--', label='H=6')
        else:
            ax.errorbar(my_list, avg_t, yerr=list(map(lambda x: 1.96 * x / gflags.FLAGS.n_runs, std_dev)),
                        fmt='g^-', label='H=10')

    ax.legend()

    # ax.set_xticklabels(map(lambda x: str(x), range()))
    ax.set_xlabel(y_label, fontsize=15)
    ax.set_ylabel('Capture Time [avg. steps]', fontsize=15)

    plt.plot()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


def plot_results():

    settings, base_name = my_exp_encoding()

    # collect info on each setting (e.g CH10S1-5G1TNSV)
    for e in settings:

        # initialize
        mip_gaps, mip_times, obj_fcn_avg, obj_fcn_std_dev, n_not_found = [], [], [], [], []

        folder_interest, subfolders = get_folder_path(e)

        # get base name depending on varying h or m
        st_base = base_name['st']
        end_base = base_name['end']
        my_list, y_label = get_label()

        for my_el in my_list:

            # print(key_l + str(my_el))
            # This is the "base_filename" for the files used inside the folder identified above
            base_filename = st_base + str(my_el) + end_base

            # init lists
            n_not_found.append(0)
            mip_gaps.append([])
            mip_times.append([])
            obj_fun_vals = []

            for i in range(1, gflags.FLAGS.n_runs + 1):
                # get number of file
                n_file = get_number_file(key_l, my_el, my_list, i)
                # retrieve data from file
                data = get_data_from_base_and_n_run(base_filename, n_file, folder_interest, subfolders)
                t = 0

                sol_time, sol_gap, sol_obj = get_from_data(data, t)

                # add to the record
                mip_times[-1].append(sol_time)
                mip_gaps[-1].append(100 * sol_gap)
                obj_fun_vals.append(sol_obj)

                if data['target'].is_captured is False:
                    n_not_found[-1] += 1

            obj_fcn_avg.append(np.mean(obj_fun_vals))
            obj_fcn_std_dev.append(np.std(obj_fun_vals))

        print(str(obj_fcn_avg) + ' ' + str(obj_fcn_std_dev))

        # all the figures are saved in the ''figs'' folder (make sure to create it)
        plot_gaps(mip_gaps, my_list, y_label, '../figs/gaps_' + e + '.png')
        plot_times(mip_times, my_list, y_label, '../figs/times_' + e + '.png')
        plot_obj(obj_fcn_avg, obj_fcn_std_dev, my_list, y_label, '../figs/obj_' + e + '.png')


def plot_obj_c_vs_d():

    solver_type = gflags.FLAGS.solver_type
    scenarios = gflags.FLAGS.exp_env

    # collect info on each setting (e.g CH10S1-5G1TNSV)
    all_mip_gaps, all_mip_times, all_obj_fcn_avg, all_obj_fcn_std_dev = {}, {}, {}, {}
    my_list, y_label = [], []

    for solver in solver_type:
        st = solver
        settings = my_exp_encoding()

        for e in settings:
            scene = scenarios[settings.index(e)]
            # initialize
            mip_gaps, mip_times, obj_fcn_avg, obj_fcn_std_dev, n_not_found = [], [], [], [], []

            folder_interest, subfolders = get_folder_path(e)

            # get base name depending on varying H or C
            st_base, end_base, key_l = get_separate_base(e)
            my_list, y_label = get_label(key_l)

            for my_el in my_list:

                # print(key_l + str(my_el))
                # This is the "base_filename" for the files used inside the folder identified above
                base_filename = st_base + str(my_el) + end_base

                # init lists
                n_not_found.append(0)
                mip_gaps.append([])
                mip_times.append([])
                obj_fun_vals = []

                for i in range(1, gflags.FLAGS.n_runs + 1):
                    # get number of file
                    n_file = get_number_file(key_l, my_el, my_list, i)
                    # retrieve data from file
                    data = get_data_from_base_and_n_run(base_filename, n_file, folder_interest, subfolders)
                    t = 0

                    sol_time, sol_gap, sol_obj = get_from_data(data, t)

                    # add to the record
                    mip_times[-1].append(sol_time)
                    mip_gaps[-1].append(100 * sol_gap)
                    obj_fun_vals.append(sol_obj)

                    if data['target'].is_captured is False:
                        n_not_found[-1] += 1

                obj_fcn_avg.append(np.mean(obj_fun_vals))
                obj_fcn_std_dev.append(np.std(obj_fun_vals))

            print(str(obj_fcn_avg) + ' ' + str(obj_fcn_std_dev))

            # all the figures are saved in the ''figs'' folder (make sure to create it)
            plot_gaps(mip_gaps, my_list, y_label, '../figs/gaps_' + e + '.png')
            plot_times(mip_times, my_list, y_label, '../figs/times_' + e + '.png')
            plot_obj(obj_fcn_avg, obj_fcn_std_dev, my_list, y_label, '../figs/obj_' + e + '.png')

            all_mip_gaps[scene, st] = mip_gaps
            all_mip_times[scene, st] = mip_times
            all_obj_fcn_avg[scene, st] = obj_fcn_avg
            all_obj_fcn_std_dev[scene, st] = obj_fcn_std_dev

    obj = {}
    for scene in scenarios:
        obj[scene] = []
        rewdC = all_obj_fcn_avg[scene, 'C']
        rewdD = all_obj_fcn_avg[scene, 'D']
        for ins in range(0, len(all_obj_fcn_avg[scene, 'C'])):
            dif_reward = rewdC[ins] - rewdD[ins]
            obj[scene].append(dif_reward)
    plot_obj_multi2(obj, my_list, y_label, '../figs/obj' + 'both' + '.png')


# --------------------------------------------------------------------------------------------------------------------
# handling info

def get_difference(list1, list2):

    my_size = len(list1)
    my_range = list(range(0, my_size))
    diff = None

    if len(list2) != my_size:
        print('Error! lists have different sizes')
        exit()
    else:
        diff = []
        for i in my_range:
            diff[i] = list1[i] - list2[i]

    return diff


# this is a generic function that retrieves the results (pickle file) of a particular experiment.
def get_data_from_base_and_n_run(base, n_run, log_path, subfolders):
    """
    :param base: the sub folder name right before the _DATE_InstanceNumber
    :param n_run: the INSTANCE number in the subfolder name
    :param log_folder: path to the main log folder containing all the runs of an experiment (e.g. ../data/CH6-14S1G1TNSV/)
    :param subfolders: the list of all the sub folders contained in log_folder
    :return:
    """
    for subfolder in subfolders:
        splitted = subfolder.split('_')
        # get basename, compare to base; compare n_run with experiment instance
        if splitted[0] == base and str(n_run).zfill(3) == splitted[2]:
            filepath = log_path + '/' + subfolder + '/global_save.txt'
            try:
                data = pickle.load(open(filepath, "rb"))
            except:
                print('Make sure your parameters are right!')
                data = None
                exit()

            return data


def get_from_data(data, t: int):

    if data['solver_data'].solver_type != 'central':
        sol_time = ext.get_last_info(data['solver_data'].solve_time[t])[1]
        sol_gap = ext.get_last_info(data['solver_data'].gap[t])[1]
        sol_obj = ext.get_last_info(data['solver_data'].obj_value[t])[1]

    else:
        sol_time = ext.get_last_info(data['solver_data'].solve_time)[1]
        sol_gap = ext.get_last_info(data['solver_data'].gap)[1]
        sol_obj = ext.get_last_info(data['solver_data'].obj_value)[1]

    return sol_time, sol_gap, sol_obj


# --------------------------------------------------------------------------------------------------------------------




# if __name__ == '__main__':
#     argv = gflags.FLAGS(sys.argv)
#     gflags.DEFINE_integer('m', gflags.FLAGS.m_list[0], 'team size')
#     gflags.DEFINE_integer('h', gflags.FLAGS.h_list[0], 'planning horizon')
#     plot_for_me()