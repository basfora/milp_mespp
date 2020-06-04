from core import extract_info as ext
from igraph import *
import os
import matplotlib.pyplot as plt
from core import sim_fun as sf


def show_me_results(md, g, name_folder: str, searchers_info: dict, deadline: int):

    s_pos, b_target = query_variables(md, searchers_info)
    plot_all_steps(g, name_folder, s_pos, b_target, deadline)


def query_variables(md, searchers_info: dict):
    """query variable X to see the optimal path"""
    S, m = ext.get_set_searchers(searchers_info)

    # save as dictionaries with searchers as keys
    x_searchers = ext.create_dict(S, None)
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

            if x_searchers[s] is None:
                x_searchers[s] = {}

            if my_var_value > 0.5:
                print('%s = %f ' % (my_var_name, my_var_value))
                x_searchers[s][(v, t)] = 1

            if t > t_max:
                t_max = t

        elif 'beta' in my_var_name and '_s' not in my_var_name:
            # print('%s %g' % (my_var_name, my_var_value))
            # remember b[0] is probability of capture
            v = int(my_var_name[5:my_var_name.find(",")])
            t = int(my_var_name[my_var_name.find(",") + 1:my_var_name.rfind("]")])
            b_target[(v, t)] = my_var_value

    # make sure x is binary
    # x_searchers = enforce_binary(x_searchers, t_max, S)
    # b_target = enforce_sum_1(b_target, t_max)

    # x_searchers[s][(v, t)]
    return x_searchers, b_target


def enforce_binary(x_searchers, t_max, S):
    """Enforce variable to be binary"""

    old_x_searchers = x_searchers

    # loop searchers
    for s in S:
        # s_var[(v, t)] = 0/1
        s_var = x_searchers.get(s)

        s_keys = s_var.keys()

        # loop through time
        for t in range(t_max + 1):

            var_value = [s_var.get(k) for k in s_keys if t == k[1]]
            v_t = [k for k in s_keys if t == k[1]]

            # find max value idx at time t
            mx = var_value.index(max(var_value))

            # everything else is zero
            for k in v_t:
                if k == v_t[mx]:
                    x_searchers[s][k] = 1
                else:
                    x_searchers[s][k] = 0

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
            vl = round(b_values[i]/my_sum, 3)

            b_target[k] = vl

    return b_target


def query_and_print_variables(md, searchers_info: dict):
    """query variable X to see the optimal path"""
    info_vector = searchers_info.keys()
    S, m = ext.get_set_searchers(info_vector)
    # V, n = ext.get_set_vertices(g)

    # save as dictionaries with searchers as keys
    x_searchers = {s_name: {} for s_name in S}
    b_target = {}   # np.zeros((n + 1, deadline + 2))

    for var in md.getVars():
        my_var_name = var.varName
        my_var_value = var.x
        print('%s %g' % (my_var_name, my_var_value))

        if 'x' in my_var_name:
            s = int(my_var_name[2])
            v = int(my_var_name[4])
            t = int(my_var_name[6])

            if my_var_value >= 0.5:
                x_searchers[s][v, t] = 1
            else:
                x_searchers[s][v, t] = 0

        elif 'beta' in my_var_name:
            # print('%s %g' % (my_var_name, my_var_value))
            # remember b[0] is probability of capture
            v = int(my_var_name[5])
            t = int(my_var_name[7])
            b_target[v, t] = my_var_value

    obj = md.getObjective()
    print(obj.getValue())

    return x_searchers, b_target


def plot_searchers_position(g, folder_name, my_layout, s_pos: dict, t: int):
    """plot results of searchers position"""
    m = len(s_pos)
    V, n = ext.get_set_vertices(g)
    S = ext.get_set_searchers(list(range(m)))[0]
    g.vs["color"] = "white"

    for s in S:
        my_searcher_pos = s_pos.get(s)
        for v in V:
            v_idx = ext.get_python_idx(v)
            my_value = my_searcher_pos.get((v, t))
            if my_value == 1:
                g.vs[v_idx]["color"] = "blue"
    name_file = folder_name + "/" + g["name"] + "_t" + str(t) + ".png"
    plot(g, name_file, layout=my_layout, figsize=(3, 3), bbox=(400, 400), margin=15, dpi=400)
    return name_file


def plot_target_belief(g, folder_name, my_layout, b_target: dict, t: int):
    """Plot target belief"""
    V, n = ext.get_set_vertices(g)

    rgba_color = (255, 0, 0, 1)
    g.vs["color"] = "white"

    for v in V:
        v_idx = ext.get_python_idx(v)
        b = b_target[v, t]
        c = b/1
        if 0.001 < c < 0.7:
            c = 0.7
        my_color = get_color_belief(rgba_color, c)
        if my_color[3] < 0.001:
            my_color = "white"
        g.vs[v_idx]["color"] = my_color
    name_file = folder_name + "/" + g["name"] + "_tgt_t" + str(t) + ".png"
    plot(g, name_file, layout=my_layout, figsize=(3, 3), bbox=(400, 400), margin=15, dpi=400)

    return name_file


def plot_all_steps(g, name_folder: str, s_pos: dict, b_target: dict, deadline: int):
    """Plot both searchers and target graph in one figure
    add title and relevant text
    save the figure in specified name folder"""
    Tau_ext = ext.get_set_ext_time(deadline)

    # TODO find out how to plot in wanted x-y coordinates
    # my_layout = g.layout("kk")
    my_layout = g.layout("tree")

    # create new folder to save figures
    if not os.path.exists(name_folder):
        os.mkdir(name_folder)
    else:
        print("Directory " + name_folder + " already exists")

    for t in Tau_ext:
        tgt_file = plot_target_belief(g, name_folder, my_layout, b_target, t)
        s_file = plot_searchers_position(g, name_folder, my_layout, s_pos, t)
        b0 = b_target.get((0, t))
        mount_frame(s_file, tgt_file, name_folder, b0, t, deadline)

    # get until core
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    # whole path
    path_now = path_this_file + "/" + name_folder
    compose_video(path_now)



def mount_frame(s_file: str, tgt_file: str, folder_name: str, b_0, t: int, deadline: int):
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    im_1 = plt.imread(path_this_file + "/" + s_file)
    im_2 = plt.imread(path_this_file + "/" + tgt_file)

    fig_1, ax_arr = plt.subplots(1, 2, figsize=(9, 5), dpi=400)

    size_title = 13
    size_subtitle = 12
    size_text = 10

    my_font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': size_subtitle,
               'horizontalalignment': 'center'}
    my_font2 = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': size_title,
                'horizontalalignment': 'center'}
    my_font3 = {'family': 'serif', 'color': 'darkgray', 'weight': 'normal', 'size': size_text,
                'horizontalalignment': 'center'}
    my_font4 = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': size_title,
                'horizontalalignment': 'center'}

    # set overall title of figure
    my_title = 'Probability of Capture: ' + str(round(b_0, 2))  # r'$\beta_o (t)$ = ' + str(round(b_0, 2))
    fig_1.text(0.5, 0.93, 'Multi-Robot Search', fontdict=my_font)
    fig_1.text(0.5, 0.88, 't = ' + str(t), fontdict=my_font4)
    fig_1.text(0.5, 0.83, my_title, fontdict=my_font2)

    # searcher positions
    ax_arr[0].imshow(im_1)
    # target belief
    ax_arr[1].imshow(im_2)

    if t == deadline:
        n_frame_per = 60
        if b_0 > 0.99:
            fig_1.text(0.5, 0.1, 'Target will be captured by deadline!', fontdict=my_font2)
        else:
            fig_1.text(0.5, 0.1, 'Probability of target capture by deadline: ' + str(int((round(b_0, 2) * 100))) + '%',
                       fontdict=my_font2)
    else:
        n_frame_per = 60
        fig_1.text(0.25, 0.05, '2 searchers, \n deterministic motion \n optimal path', fontdict=my_font3)
        # fig_1.text(0.75, 0.05, '1 target, \nrandom motion \n  known initial position', fontdict=my_font3)
        # fig_1.text(0.75, 0.05, '1 target, \nrandom motion \n  uniform initial distribution', fontdict=my_font3)
        fig_1.text(0.75, 0.05, '1 target, \nstatic motion \n  known initial position', fontdict=my_font3)

    # take out axis stuff
    for k in range(0, 2):
        ax_arr[k].set_xticklabels([])
        ax_arr[k].set_xticks([])
        ax_arr[k].set_yticklabels([])
        ax_arr[k].set_yticks([])
        ax_arr[k].axis('off')

    my_format = ".png"

    # save the frame
    # change n_start to 140 for complete video 140  # n_frame_per * 3
    n_start = 0
    for i in range(n_frame_per):
        frame_num = n_start + i + t * n_frame_per
        frame_string = str(frame_num)
        frame_string = frame_string.rjust(4, "0")
        fname = path_this_file + "/" + folder_name + "/" + "frame_" + frame_string + my_format
        # plt.figure(figsize=(4, 8), dpi=400)
        plt.savefig(fname, facecolor=None, edgecolor=None,
                    orientation='landscape', papertype=None,
                    transparent=True)


def compose_video(path_now: str):
    """Use plots as frames and make a short video"""
    command_to_run = "ffmpeg -r 20 -f image2 -i " + path_now + "/frame_%04d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p "\
                     + path_now + "/a_no_sync.mp4 -y"
    os.system(command_to_run)

# --------------------------------------------------------------------------------------------------------------------
# simulation results


def create_my_folder(today_run=0):
    """Create directory if it doesnt exist, return the name of the directory (not the path)"""

    name_folder, whole_path = ext.get_name_folder(today_run)

    print(whole_path)
    # create new folder to save figures
    if not os.path.exists(whole_path):
        os.mkdir(whole_path)
    else:
        print("Directory " + name_folder + " already exists")
    return name_folder


def create_my_new_folder(m_searcher, g, today_run=0, solver_type='central', zeta=None, capture_range=0,
                         horizon=None, day_start=None):

    name_folder, whole_path = ext.get_name_code_folder(today_run, m_searcher, g, solver_type, zeta, capture_range,
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
    name_folder, whole_path = ext.get_name_code_folder(today_run, m, g, solver_type, zeta, capture_range, horizon,
                                                       day_start)

    # print(whole_path)
    # create new folder to save figures
    if not os.path.exists(whole_path):
        os.mkdir(whole_path)
    else:
        print("Directory " + name_folder + " already exists")
    return name_folder


def plot_and_show_sim_results(belief, target, searchers, sim_data, folder_name):
    """Get information of the simulation from the classes
    turn into short video"""

    # graph file and layout
    g, my_layout = retrieve_graph(sim_data)

    # data folder path
    folder_path = ext.get_whole_path(folder_name)

    # time info
    max_time = ext.get_last_info(target.stored_v_true)[0]
    Tau_ext = ext.get_set_ext_time(max_time)

    # capture info
    capture_time = target.capture_time
    captor = sf.find_captor(searchers)
    vertex_cap = target.capture_v
    # pack
    capture_info = [capture_time, vertex_cap, captor]

    print("Starting plots")
    for t in Tau_ext:
        # get belief at that time
        b_target = belief.stored[t]
        # get POC(t)
        b0 = b_target[0]
        # plot belief
        tgt_file = plot_target_belief2(g, folder_path, my_layout, b_target, t)
        # plot searchers and true target position
        s_file = plot_searchers_and_target(g, folder_path, my_layout, target, searchers, t)
        # assemble it nicely and make copies
        mount_sim_frame(s_file, tgt_file, folder_path, b0, t, capture_info)

    # get until multiRobotTargetSearch/data/name_folder
    print("Composing video")
    # compose short video
    compose_video(folder_path)
    print("Video is done")
    delete_frames(folder_path)
    print("Frames were deleted")
    return


def plot_sim_results(belief, target, searchers, sim_data, folder_name):
    """Get information of the simulation from the classes
    turn into short video"""

    g, my_layout = retrieve_graph(sim_data)

    # data folder path
    folder_path = ext.get_whole_path(folder_name)

    # time info
    max_time = ext.get_last_info(target.stored_v_true)[0]
    Tau_ext = ext.get_set_ext_time(max_time)

    # capture info
    capture_time = target.capture_time
    captor = sf.find_captor(searchers)
    vertex_cap = target.capture_v
    # pack
    capture_info = [capture_time, vertex_cap, captor]

    print("Starting plots")
    for t in Tau_ext:
        # get belief at that time
        b_target = belief.stored[t]
        # get POC(t)
        b0 = b_target[0]
        # plot belief
        tgt_file = plot_target_belief2(g, folder_path, my_layout, b_target, t)
        # plot searchers and true target position
        s_file = plot_searchers_and_target(g, folder_path, my_layout, target, searchers, t)
        # assemble it nicely
        mount_sim_frame(s_file, tgt_file, folder_path, b0, t, capture_info, 1)

    delete_frames(folder_path, 'G')
    print("Frames were deleted")

    return


def delete_frames(folder_path: str, key_name='frame'):
    """Delete frames used to make the video to save space"""

    for filename in os.listdir(folder_path):
        if filename.startswith(key_name) and filename.endswith('.png'):
            my_file = folder_path + "/" + filename
            os.remove(my_file)
    return


def plot_searchers_and_target(g, folder_path, my_layout, target, searchers, t):
    """plot results of searchers position
    and true position of the target"""

    m = len(list(searchers.keys()))
    S = ext.get_set_searchers(list(range(m)))[0]
    g.vs["color"] = "white"

    for s_id in S:
        s = searchers[s_id]
        my_searcher_vertex = s.path_taken[t]
        v_idx = ext.get_python_idx(my_searcher_vertex)
        g.vs[v_idx]["color"] = "blue"

        # plot target
        v_target = target.stored_v_true[t]
        v_t_idx = ext.get_python_idx(v_target)
        if target.capture_time == t:
            g.vs[v_t_idx]["color"] = "green"
        else:
            g.vs[v_t_idx]["color"] = "red"

    name_file = folder_path + "/" + g["name"] + "_t" + str(t) + ".png"
    plot(g, name_file, layout=my_layout, figsize=(3, 3), bbox=(400, 400), margin=15, dpi=400)
    plt.close()
    return name_file


def mount_sim_frame(s_file: str, tgt_file: str, folder_path: str, b_0, t: int, capture_info: list, n_frame_per=60):

    # unpack capture info
    capture_time, vertex_cap, captor = capture_info

    im_1 = plt.imread(s_file)
    im_2 = plt.imread(tgt_file)

    fig_1, ax_arr = plt.subplots(1, 2, figsize=(9, 5), dpi=400)

    size_title = 13
    size_subtitle = 12
    size_text = 10

    my_font = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': size_subtitle,
               'horizontalalignment': 'center'}
    my_font2 = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': size_title,
                'horizontalalignment': 'center'}
    my_font3 = {'family': 'serif', 'color': 'darkgray', 'weight': 'normal', 'size': size_text,
                'horizontalalignment': 'center'}
    my_font4 = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': size_title,
                'horizontalalignment': 'center'}

    # set overall title of figure
    my_title = 'Probability of Capture: ' + str(round(b_0, 2))  # r'$\beta_o (t)$ = ' + str(round(b_0, 2))
    fig_1.text(0.5, 0.93, 'Multi-Robot Search Simulation', fontdict=my_font)
    fig_1.text(0.5, 0.88, 't = ' + str(t), fontdict=my_font4)
    fig_1.text(0.5, 0.83, my_title, fontdict=my_font2)

    # searcher positions
    ax_arr[0].imshow(im_1)
    # target belief
    ax_arr[1].imshow(im_2)

    if t == capture_time:
        text_cap = 'Target detected at vertex ' + str(vertex_cap) + ' by searcher ' + str(captor)
        fig_1.text(0.5, 0.1, text_cap, fontdict=my_font2)
    else:
        fig_1.text(0.30, 0.05, 'True position of searchers and target', fontdict=my_font3)
        fig_1.text(0.75, 0.05, 'Belief of target location', fontdict=my_font3)

    # take out axis stuff
    for k in range(0, 2):
        ax_arr[k].set_xticklabels([])
        ax_arr[k].set_xticks([])
        ax_arr[k].set_yticklabels([])
        ax_arr[k].set_yticks([])
        ax_arr[k].axis('off')

    my_format = ".png"

    # save the frame
    # change n_start to 140 for complete video 140  # n_frame_per * 3
    n_start = 0
    for i in range(n_frame_per):
        frame_num = n_start + i + t * n_frame_per
        frame_string = str(frame_num)
        frame_string = frame_string.rjust(4, "0")

        fname = folder_path + "/" + "frame_" + frame_string + my_format

        # plt.figure(figsize=(4, 8), dpi=400)
        plt.savefig(fname, facecolor=None, edgecolor=None,
                    orientation='landscape', papertype=None,
                    transparent=True)
        plt.clf()

    plt.close('all')


def plot_target_belief2(g, folder_path, my_layout, b_target: dict, t: int):
    """Plot target belief"""
    V, n = ext.get_set_vertices(g)

    rgba_color = (255, 0, 0, 1)
    g.vs["color"] = "white"

    for v in V:
        v_idx = ext.get_python_idx(v)
        b = b_target[v]

        if b <= 0.005:
            my_color = "white"
        else:
            if 0.005 < b <= 0.1:
                c = 0.1
            elif 0.1 < b <= 0.2:
                c = 0.2
            elif 0.2 < b <= 0.3:
                c = 0.3
            elif 0.3 < b <= 0.4:
                c = 0.4
            elif 0.4 < b <= 0.5:
                c = 0.5
            elif 0.5 < b <= 0.6:
                c = 0.6
            elif 0.6 < b <= 0.7:
                c = 0.7
            elif 0.7 < b <= 0.8:
                c = 0.8
            else:
                c = b/1
            my_color = get_color_belief(rgba_color, c)

        g.vs[v_idx]["color"] = my_color
    name_file = folder_path + "/" + g["name"] + "_tgt_t" + str(t) + ".png"
    plot(g, name_file, layout=my_layout, figsize=(3, 3), bbox=(400, 400), margin=15, dpi=400)
    plt.close()

    return name_file


def get_color_belief(rgb_color, ratio=0.5):
    """ Return color proportional to belief (darker is stronger)"""
    red, green, blue, alpha = rgb_color
    return red, green, blue, (alpha * ratio)


def retrieve_graph(sim_data):
    # graph file and layout
    g = sim_data.g
    if 'grid' in g['name']:
        my_layout = g.layout("grid")
    else:
        my_layout = g.layout("kk")

    return g, my_layout







