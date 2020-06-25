import os

from igraph import plot
from matplotlib import pyplot as plt

from core import extract_info as ext
from core import milp_fun as mf


# -----------------------------------------------------------------------------------------
# Plot functions for MILP MESPP
# -----------------------------------------------------------------------------------------
from core.extract_info import retrieve_graph


def show_me_results(md, g, name_folder: str, deadline: int):

    x_searchers, b_target = mf.query_variables(md)
    plot_all_steps(g, name_folder, x_searchers, b_target, deadline)


def plot_all_steps(g, name_folder: str, x_searchers: dict, b_target: dict, deadline: int):
    """Plot both searchers and target graph in one figure
    add title and relevant text
    save the figure in specified name folder"""
    Tau_ext = ext.get_set_ext_time(deadline)

    # my_layout = g.layout("kk")
    my_layout = g.layout("tree")

    # create new folder to save figures
    if not os.path.exists(name_folder):
        os.mkdir(name_folder)
    else:
        print("Directory " + name_folder + " already exists")

    for t in Tau_ext:
        tgt_file = plot_target_belief(g, name_folder, my_layout, b_target, t)
        s_file = plot_searchers_position(g, name_folder, my_layout, x_searchers, t)
        b0 = b_target.get((0, t))
        mount_frame_mespp(s_file, tgt_file, name_folder, b0, t, deadline)

    # get until core
    path_this_file = os.path.dirname(os.path.abspath(__file__))
    # whole path
    path_now = path_this_file + "/" + name_folder
    compose_video(path_now)


def plot_searchers_position(g, folder_name, my_layout, x_searchers: dict, t: int):
    """plot results of searchers position"""

    m = ext.get_m_from_xs(x_searchers)

    V, n = ext.get_set_vertices(g)
    S = ext.get_set_searchers(m)[0]
    g.vs["color"] = "white"

    for s in S:
        for v in V:
            my_value = x_searchers.get((s, v, t))

            if my_value == 1:
                v_idx = ext.get_python_idx(v)
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


def mount_frame_mespp(s_file: str, tgt_file: str, folder_name: str, b_0, t: int, deadline: int):
    """Mount a nice frame to show searchers position and target belief at each time step
    1st line: Multi-Robot Search Simulation
    2nd line: t = [time-step]
    3rd line: Probability of Capture [b_c]
    LEFT: True position of searchers (blue) and target (red)
    RIGHT: Belief of target location (shades of red)
    """
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


def get_color_belief(rgb_color, ratio=0.5):
    """ Return color proportional to belief (darker is stronger)"""
    red, green, blue, alpha = rgb_color
    return red, green, blue, (alpha * ratio)


# -----------------------------------------------------------------------------------------
# Plot functions for MILP MESPP RISK-AWARE
# -----------------------------------------------------------------------------------------
def compose_and_clean(path_to_folder: str):
    compose_video(path_to_folder)
    delete_frames(path_to_folder)


def compose_video(path_to_folder: str):
    """Use plots as frames and make a short video"""
    # print(path_now)
    print("Composing video")
    command_to_run = "ffmpeg -r 20 -f image2 -i " + path_to_folder + \
                     "/frame_%04d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p " \
                     + path_to_folder + "/a_no_sync.mp4 -y"
    os.system(command_to_run)
    print("Video is done")


def delete_frames(path_to_folder: str, key_name='frame'):
    """Delete frames used to make the video to save space"""

    for filename in os.listdir(path_to_folder):
        if filename.startswith(key_name) and filename.endswith('.png'):
            my_file = path_to_folder + "/" + filename
            os.remove(my_file)

    print("Frames were deleted")
    return


def define_fonts():
    """Define dictionary with fonts for plotting"""

    n_fts = 4

    my_font = ext.create_dict(list(range(0, n_fts)), None)
    my_sizes = ext.create_dict(list(range(0, n_fts)), 10)

    # font and sizes
    my_sizes[0] = 13
    my_sizes[1] = 12
    my_sizes[2] = 11
    my_sizes[3] = 9

    # title - red and bold
    my_font[0] = {'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': my_sizes[0],
                  'horizontalalignment': 'center'}

    # subtitle - dark blue and bold
    my_font[1] = {'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': my_sizes[1],
                  'horizontalalignment': 'center'}
    # regular text - gray
    my_font[2] = {'family': 'serif', 'color': 'darkgray', 'weight': 'normal', 'size': my_sizes[3],
                  'horizontalalignment': 'center'}

    # regular text - black
    my_font[3] = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': my_sizes[3],
                  'horizontalalignment': 'center'}

    return my_font


def place_image(im, ax_arr, my_subs: list):
    """ place image in subplot and take out ticks and stuff"""

    if len(my_subs) > 1:
        # plt.tight_layout()
        new_left = 0.1
        new_right = 0.9
        new_bottom = 0.1
        new_top = 0.9
        plt.subplots_adjust(wspace=0.1, hspace=0, left=new_left, right=new_right, bottom=new_bottom, top=new_top)

        for k in my_subs:
            ax_arr[k].imshow(im[k])
            ax_arr[k].set_xticklabels([])
            ax_arr[k].set_xticks([])
            ax_arr[k].set_yticklabels([])
            ax_arr[k].set_yticks([])
            ax_arr[k].axis('off')
    else:

        ax_arr.imshow(im[0])

        ax_arr.set_xticklabels([])
        ax_arr.set_xticks([])
        ax_arr.set_yticklabels([])
        ax_arr.set_yticks([])
        ax_arr.axis('off')

    return


def mount_frame(path_and_fig, t: int, my_words: dict, n_sub=1, video=False):
    """Mount frame for video
    :path_and_fig: path+name(s) of figure(s)
    :t: time step
    :my_words: dict with 3 keys in order my_title, unit of time, subtitle
    :n_sub: number of subplots"""

    # ----------------
    my_font = define_fonts()
    # -----------------

    # how many subplots
    my_subs = list(range(0, n_sub))

    # create figure with subplots
    fig_1, ax_arr = plt.subplots(1, n_sub, figsize=(9, 5), dpi=600)

    # retrieve graph plots as images
    im = {}
    if n_sub == 1:
        if isinstance(path_and_fig, str):
            im[0] = plt.imread(path_and_fig)
        else:
            im[0] = plt.imread(path_and_fig[2])

    else:
        for i in my_subs:
            im[i] = plt.imread(path_and_fig[i])

    # -------------------
    # plot text
    # insert time step
    my_words[0]['text'] = my_words[0]['text'] + str(t)

    # title and subtitle
    for line in range(0, 2):
        my_text = my_words[line]['text']
        x, y = my_words[line]['xy']

        fig_1.text(x, y, my_text, fontdict=my_font[line])

    if n_sub == 3:
        for col in range(1, 5):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all cols
            idx = 1
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

        for col in range(5, 11):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all sub cols
            idx = 2
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

    # labels
    my_hazard_labels(fig_1)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    # # -------------------

    # take out axis stuff
    place_image(im, ax_arr, my_subs)
    # -------------------

    # save in folder
    frame_idx = t
    save_frame(path_and_fig, frame_idx, video)


def save_frame(path_and_fig, frame_idx, video=False):
    path_to_folder = os.path.dirname(path_and_fig[0])

    if video:
        save_copied_frames(path_to_folder, frame_idx)
    else:
        save_one_frame(path_to_folder, frame_idx)

    return


def save_one_frame(path_to_folder, frame_idx, key='hazard'):
    my_format = ".png"

    frame_num = frame_idx
    frame_string = str(frame_num)
    frame_string = frame_string.rjust(4, "0")

    fname = path_to_folder + "/" + key + "_" + frame_string + my_format

    # plt.figure(figsize=(4, 8), dpi=400)
    plt.savefig(fname, facecolor=None, edgecolor=None,
                orientation='landscape', papertype=None,
                transparent=True)


def save_copied_frames(path_to_folder: str, frame_idx: int, n_frames_per=60):
    """Multiply frames for video"""

    my_format = ".png"

    # save the frame
    # change n_start to 140 for complete video 140  # n_frame_per * 3
    n_start = 0
    for i in range(n_frames_per):
        frame_num = n_start + i + frame_idx * n_frames_per
        frame_string = str(frame_num)
        frame_string = frame_string.rjust(4, "0")

        fname = path_to_folder + "/" + "frame_" + frame_string + my_format

        # plt.figure(figsize=(4, 8), dpi=400)
        plt.savefig(fname, facecolor=None, edgecolor=None,
                    orientation='landscape', papertype=None,
                    transparent=True)


def my_hazard_labels(fig_1):

    levels = [1, 2, 3, 4, 5]
    # level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    # level_color = ['green', 'blue', 'yellow', 'orange', 'red']
    level_color = color_levels(True)
    level_label = label_levels()

    my_font = {}

    x, y = 0.31, 0.1

    for i in levels:

        my_font[i] = {'family': 'serif', 'color': level_color[i], 'weight': 'bold', 'size': 11,
                'horizontalalignment': 'center'}

        fig_1.text(x, y, level_label[i], fontdict=my_font[i])

        x = x + 0.1

    return fig_1


def empty_my_words(n: int):
    my_words = ext.create_dict(list(range(0, n)), None)

    for i in range(0, n):
        my_words[i] = {'text': '', 'xy': (0.0, 0.0)}

    return my_words


def color_levels(normalized=False):
    """Change colors here"""

    level = ext.create_dict([1, 2, 3, 4, 5], None)

    # (R, G, B)
    # yellow = (1, 1, 0)
    # orange =

    # green
    level[1] = (60, 180, 60)
    # yellow-green
    level[2] = (200, 200, 30)
    # yellow
    level[3] = (240, 215, 40)
    # orange
    level[4] = (250, 120, 50)
    # red
    level[5] = (255, 30, 30)

    if normalized is True:
        for k in level.keys():
            r = level[k][0]
            g = level[k][1]
            b = level[k][2]
            level[k] = (r/255, g/255, b/255)
    return level


def label_levels():
    labels = ext.create_dict(list(range(1, 6)), '')

    labels[1] = 'Low'
    labels[2] = 'Moderate'
    labels[3] = 'High'
    labels[4] = 'Very High'
    labels[5] = 'Extreme'

    return labels


def match_level_color(my_level: int):
    """Define color for levels"""

    colors = color_levels(True)

    my_color = set_vertex_color(colors[my_level])

    return my_color


def set_vertex_color(my_color: tuple):

    alpha = 1
    red, green, blue = my_color
    return red, green, blue, alpha


# -----------------------------------------------------------------------------------------
# Plot functions for MILP MESPP MOVED
# -----------------------------------------------------------------------------------------


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
    captor = ext.find_captor(searchers)
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
    pf.compose_video(folder_path)
    print("Video is done")
    pf.delete_frames(folder_path)
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
    captor = ext.find_captor(searchers)
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