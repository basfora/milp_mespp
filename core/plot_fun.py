import matplotlib.pyplot as plt
import os
from core import extract_info as ext
# from core import extract_info as ext
# from igraph import *


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