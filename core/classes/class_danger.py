from core import extract_info as ext
from core.classes.class_hazard import MyHazard
from core import plot_fun as pf
import pickle
from igraph import *
import time


class MyDanger:
    """Define danger
    using hazards: fire (f), smoke (z)
    Bayesian network: fire + smoke => danger
    conditional probability table"""

    # TODO enable online option (update at each t)

    def __init__(self, g, deadline, plot_for_me=False, f=None, z=None, h_0=None):

        # save graph
        self.g_name = ""
        self.g = None
        # name of folder for this sim: eg smoke_G9V_grid_date#_run#
        self.folder_name = ""
        # whole path + /name_folder
        self.whole_path = ""

        self.plot = plot_for_me

        # vertices
        self.n = 0
        self.V = []

        # time related info
        self.tau = 0
        self.T = []
        self.T_ext = []

        # hazards
        self.type = 'danger'
        # smoke
        self.z = z
        # fire
        self.f = f

        # P(danger|f, z) dictionaries
        # p[(i, j)] = []
        self.p_zf = None
        # p[(i, j, k)] = value
        self.p_zfd = None

        # values throughout time
        # probability of each danger level: eta (v, t, k) = prob
        self.eta = dict()
        # probable danger level, highest eta: H (v, t) =
        self.H = dict()

        # ---------------------
        # pre defined parameters
        # ---------------------
        # danger levels
        self.levels = [1, 2, 3, 4, 5]
        self.n_levels = len(self.levels)
        self.level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        self.level_color = ['green', 'blue', 'yellow', 'orange', 'red']
        # ----------------------

        self.simulate(g, deadline, h_0)

    def simulate(self, g, deadline, h_0):
        """Simulate Default"""

        self.structure(g, deadline)
        self.init_default(h_0)
        if self.check_sum_1():
            self.assign_eta()
            # self.print_levels()
            # plotting settings
            if self.plot:
                frames, video, together = True, True, True
                print(self.get_value_v(1))
                self.plot_evolution(frames, video, together)

        else:
            exit()
    # --------------------------
    def print_levels(self):

        for t in self.T_ext:
            all_levels = ext.create_dict(self.V, [])
            for v in self.V:

                z = self.z.get_value_vt(v, t)
                f = self.f.get_value_vt(v, t)
            print('-------------------')
            print('t = ' + str(t))
            d = self.get_H_t(t)
            print(d)


            # print(all_levels)

    # called on init
    def structure(self, g, deadline):

        V, n = ext.get_set_vertices(g)
        self.n = n
        self.V = V

        self.g_name = g["name"]
        self.g = g

        T = ext.get_set_time(deadline)
        T_ext = ext.get_set_ext_time(deadline)
        self.tau = deadline
        self.T = T
        self.T_ext = T_ext

        # create dicts for eta, eta_level
        self.create_empty_dicts()

    def init_default(self, h_0):

        g = self.g
        deadline = self.tau

        # get hazards
        self.simulate_hazards(g, deadline, h_0)

        # get cond probability matrix (dict form)
        self.load_prob_dict()

    def create_empty_dicts(self):

        for v in self.V:

            # danger -- float
            # eta(v, t)
            self.eta[v] = {}

            # hazard level -- int
            # z_level(v,t)
            self.H[v] = {}

            for t in self.T:
                self.eta[v][t] = {}
                self.H[v][t] = []

    # --------------------------
    # simulate fire and smoke values for each (v, t)
    def simulate_hazards(self, g, deadline, h_0):
        """Simulate and save smoke and fire if they were not given on init"""

        self.make_hazard(g, deadline, 'smoke', h_0)
        self.make_hazard(g, deadline, 'fire', h_0)

    def make_hazard(self, g, deadline, what: str, h_0=None):
        """ Simulate smoke and fire
        don't save the data, just evolve for now"""

        h0 = self.get_initial_hazard(h_0, what)
        h = MyHazard(g, deadline, what, h0)
        h.simulate(4)

        if what == 'fire' and self.f is None:
            self.f = h
        elif what == 'smoke' and self.z is None:
            self.z = h
        else:
            print('Only smoke and fire defined at this time.')
            exit()

    # --------------------------
    # conditional probability functions
    def load_prob_dict(self, default=True):
        """load default conditional matrix"""

        self.create_empty_tables()

        if default is True:
            # retrieve values from the
            self.set_values_table()
            self.organize_table()

    def create_empty_tables(self):
        self.p_zf = ext.create_2tuple_keys(self.z.levels, self.f.levels)
        self.p_zfd = ext.create_3tuple_keys(self.z.levels, self.f.levels, self.levels)

    def set_values_table(self):
        """hij --> [smoke level|fire level] = [k1 k2 k3 k4 k5]"""

        Mzf = self.p_zf

        #   (z, f)
        Mzf[(1, 1)] = [0.96, 0.01, 0.01, 0.01, 0.01]

        Mzf[(2, 1)] = [0.01, 0.96, 0.01, 0.01, 0.01]
        Mzf[(2, 2)] = [0.02, 0.48, 0.48, 0.01, 0.01]

        Mzf[(3, 1)] = [0.02, 0.48, 0.48, 0.01, 0.01]
        Mzf[(3, 2)] = [0.02, 0.48, 0.48, 0.01, 0.01]
        Mzf[(3, 3)] = [0.01, 0.25, 0.49, 0.24, 0.01]

        Mzf[(4, 1)] = [0.01, 0.25, 0.49, 0.24, 0.01]
        Mzf[(4, 2)] = [0.01, 0.10, 0.30, 0.50, 0.09]
        Mzf[(4, 3)] = [0.01, 0.08, 0.27, 0.55, 0.09]
        Mzf[(4, 4)] = [0.01, 0.08, 0.27, 0.55, 0.09]

        Mzf[(5, 1)] = [0.01, 0.01, 0.01, 0.47, 0.50]
        Mzf[(5, 2)] = [0.01, 0.01, 0.01, 0.40, 0.57]
        Mzf[(5, 3)] = [0.01, 0.01, 0.01, 0.35, 0.62]
        Mzf[(5, 4)] = [0.01, 0.01, 0.01, 0.25, 0.72]
        Mzf[(5, 5)] = [0.01, 0.01, 0.01, 0.15, 0.82]

        # p_zf[(i, j)] = [p_l1 p_l2 p_l3 p_l4 p_l5]
        self.p_zf = Mzf

        return Mzf

    def organize_table(self):
        """Build matrix structure ijk
        i -> smoke levels
        j -> fire levels
        k -> danger levels"""

        # smoke (hazard 1)
        for i in self.z.levels:
            # fire (hazard 2)
            for j in self.f.levels:

                # check if it's empty
                if self.no_eta(i, j):
                    continue
                else:
                    # danger levels
                    for k in self.levels:
                        k_idx = ext.get_python_idx(k)
                        self.p_zfd[(i, j, k)] = self.p_zf[(i, j)][k_idx]

        # dictionary (i, j, k)
        return self.p_zfd

    def check_sum_1(self):
        """Check if fire is never higher than smoke, change it if necessary"""

        # loop through smoke levels
        for i in self.z.levels:

            # loop through fire levels
            for j in self.f.levels:

                if self.no_eta(i, j):
                    continue
                else:
                    my_sum = 0

                    # loop through danger levels
                    for k in self.levels:
                        # retrieve value
                        p = self.get_eta(i, j, k)
                        my_sum += p

                    if round(my_sum, 6) != 1:
                        print('Issue found in (i,j) = (' + str(i) + ',' + str(j) + ')')
                        print('Sum = ' + str(my_sum))
                        print(self.get_row_eta(i, j))
                        return False
        return True

    def assign_eta(self):
        """ Assign value for eta, H at each v and t
         according to the conditional probability table"""

        for v in self.V:
            for t in self.T_ext:
                i = self.z.get_value_vt(v, t, 0)
                j = self.f.get_value_vt(v, t, 0)

                # make sure smoke >= fire
                i = self.check_dependency(i, j)

                # map smoke and fire levels to danger levels probability
                row_eta = self.get_row_eta(i, j)

                # get highest probability level of danger
                high_value, H = ext.get_highest(row_eta, self.levels)

                self.eta[v][t] = row_eta
                self.H[v][t] = [high_value, H]

    # ------------------
    # extract info (doesn't change anything on class, just retrieve info)
    def get_row_eta(self, i: int, j: int):
        """Get the probability for each danger level, given smoke i and fire j levels """

        list_prob = ext.create_dict(self.levels, None)

        for k in list_prob.keys():
            list_prob[k] = self.get_eta(i, j, k)

        return list_prob

    def no_eta(self, i: int, j: int):

        if not self.p_zf[(i, j)]:
            return True
        else:
            return False

    def get_eta(self, i: int, j: int, k: int):
        """Get the probability for danger level k, given smoke i and fire j levels
        i: smoke level
        j: fire level
        k: danger level"""

        my_p = self.p_zfd[(i, j, k)]

        return my_p

    # get after assigned
    def get_value_vt(self, v: int, t: int, op=0):
        """Retrieve : H(v, t) (likely level, highest eta) or eta(v,t) for all levels"""

        if op == 0:
            # integer
            value = self.H[v][t]
        else:
            # dict with 5 probabilities
            value = self.eta[v][t]

        return value

    def get_danger_t(self, t: int, op=0):
        """Retrieve at time t: likely level (highest eta) or eta(t) for all vertices"""

        value = {}
        for v in self.V:
            if op == 0:
                my_eta = self.H[v][t]
            else:
                my_eta = self.eta[v][t]

            value[v] = my_eta

        # print(z)
        return value

    def get_H_t(self, t: int):

        # [v: [value, level]]
        my_dict = self.get_danger_t(t, 0)

        value = {}
        for v in self.V:
            value[v] = my_dict[v][1]

        # print(z)
        return value

    def get_value_v(self, v: int, op=0):
        """Retrieve for vertex v: H(v) or eta (v) for all times """

        eta = {}
        for t in self.T:
            if op == 0:
                my_eta = self.H[v][t]
            else:
                my_eta = self.eta[v][t]
            eta[t] = my_eta

        # print(eta)
        return eta

    def get_value_all(self, op=0):
        """Retrieve H or eta for all v,t"""

        if op == 0:
            eta = self.H
        else:
            eta = self.eta

        print(eta)
        return eta

    # match level value with name and color
    def get_level_name(self, value: int):
        idx = self.levels.index(value)
        name = self.level_label[idx]

        return name

    def get_level_value(self, my_name: str):
        idx = self.level_label.index(my_name)
        value = self.levels[idx]

        return value

    def get_level_color(self, my_level):

        idx = None

        if isinstance(my_level, int):
            idx = ext.get_python_idx(my_level)

        elif isinstance(my_level, str):
            id_aux = self.get_level_value(my_level)
            idx = ext.get_python_idx(id_aux)
        else:
            return idx

        my_color = self.level_color[idx]

        return my_color

    # --------------------
    # save
    def save_data(self):
        """Save danger and hazard data into a pickle file"""
        # name the pickle file
        self.create_folder()
        file_name = 'danger_save.txt'
        full_path = self.whole_path + "/" + file_name

        my_pickle = open(full_path, "wb")
        pickle.dump(self, my_pickle)
        my_pickle.close()

        print("Data saved in: ", self.folder_name)
        return

    def create_folder(self, i=1):
        """Create folder inside /data"""

        self.make_folder(i)

        my_flag = True
        while my_flag:

            if not os.path.exists(self.whole_path):
                os.mkdir(self.whole_path)
                my_flag = False
            else:
                i += 1
                self.make_folder(i)
        return

    def make_folder(self, i=1):
        my_date = ext.get_name_folder(i)[0]
        my_graph = (self.g_name.split('.'))[0]

        name_folder = self.type + '_' + my_graph + '_' + my_date
        whole_path = ext.get_whole_path(name_folder, 'data')

        # name of this folder, eg smoke_G9V_grid_date#_run#
        self.folder_name = name_folder
        # whole path + /name_folder
        self.whole_path = whole_path

    # --------------------
    # plot evolution
    def plot_evolution(self, frames=False, video=False, together=False):

        # save data first
        self.save_data()

        if together:
            n_subplots = 3
        else:
            n_subplots = 1

        path_and_fig = ext.create_dict(list(range(0, n_subplots)), None)

        for t in self.T_ext:

            path_and_fig[2] = self.get_and_plot(t, 'danger')

            if together:
                path_and_fig[0] = self.get_and_plot(t, 'smoke')
                path_and_fig[1] = self.get_and_plot(t, 'fire')

            if frames:
                # plot frames
                my_words = self.frame_details(together)

                # file name, time step and labels
                pf.mount_frame(path_and_fig, t, my_words, n_subplots, video)

        if frames:
            if video:
                # compose short video
                pf.compose_and_clean(self.whole_path)
            else:
                # pf.delete_frames(self.whole_path, self.type)
                if together:
                    pf.delete_frames(self.whole_path, 'smoke')
                    pf.delete_frames(self.whole_path, 'fire')

    def get_and_plot(self, t, var='danger'):

        if var == 'smoke':
            var_t = self.z.get_value_t(t)

        elif var == 'fire':
            var_t = self.f.get_value_t(t)
        else:
            var_t = self.get_H_t(t)
            d = self.get_H_t(t)
            print('t = ' + str(t))
            print(d)

        plot_str = self.plot_graph(var_t, t, var)

        return plot_str

    def plot_graph(self, d_t: dict, t: int, var):
        """z_t: dict z[v] v= 1,...n"""

        folder_path = self.whole_path
        # if var == 'danger':
        #     d = self.get_H_t(t)
        #     print(d)
        #     print('name_file t_' + str(t))

        for v in self.V:
            v_idx = ext.get_python_idx(v)

            my_level = d_t[v]
            my_color = pf.match_level_color(my_level)

            # if v == 1:
            #
            #     print('t = ' + str(t))
            #     print('color: ' + str(my_color))
            #    print('level = ' + str(d_t[v]))

            # assign color
            self.g.vs[v_idx]["color"] = my_color

        name_file = folder_path + "/" + var + "_t" + str(t) + ".png"

        time.sleep(0.1)

        plot(self.g, name_file, layout='grid', figsize=(5, 5), bbox=(600, 600), margin=20, dpi=400, vertex_size=40)

        return name_file

    def make_video(self):
        # save data first
        self.save_data()

        for t in self.T_ext:
            z_t = self.get_danger_t(t)
            hazard_t = self.plot_graph(z_t, t)
            my_words = self.frame_details()
            # file name, time step and labels
            pf.mount_frame(hazard_t, t, my_words, 1, True)

        # compose short video
        pf.compose_and_clean(self.whole_path)

    # --------------------
    # static methods
    @staticmethod
    def check_dependency(z_level: int, f_level: int):
        """Make sure smoke level is less or equal to fire level"""

        if z_level < f_level:
            z_level = f_level

        return z_level

    @staticmethod
    def get_initial_hazard(h0, hazard: str):

        # check if there is a initial hazard input
        if h0 is not None:
            if hazard == 'smoke':
                h_0 = h0['z']
            else:
                h_0 = h0['f']
        else:
            h_0 = None

        return h_0

    @staticmethod
    def frame_details(together=False):
        """my_words
        [0] title
        [1] subtitle

        [2]-[4] cols (1, 2, 3)
        [5]-[7] sub cols (1, 2, 3)"""

        # title , time step, subtitle

        name_hazard0 = 'Danger'

        if together is False:

            my_words = pf.empty_my_words(2)

            my_words[0]['text'] = name_hazard0 + ' Evolution , t = '
            my_words[1]['text'] = 'from fire and smoke'

            my_words[0]['xy'] = (0.5, 0.93)
            my_words[1]['xy'] = (0.5, 0.88)

        else:
            my_words = pf.empty_my_words(11)
            name_hazard1 = 'Smoke'
            name_hazard2 = 'Fire'

            my_words[0]['text'] = 'Linear Evolution Model, t = '
            my_words[1]['text'] = ''
            my_words[2]['text'] = name_hazard1
            my_words[3]['text'] = name_hazard2
            my_words[4]['text'] = name_hazard0

            my_words[5]['text'] = r'$z(t) = \xi t + z(0)$'
            my_words[6]['text'] = r'$f(t) = \iota t + f(0)$'
            my_words[7]['text'] = r'$H(t) \propto \eta_{max} (t)$'

            my_words[8]['text'] = r'$\xi$' + '~' + r'(0.3, 0.1$^2$)'
            my_words[9]['text'] = r'$\iota$' + '~' + r'(0.2, 0.05$^2$)'
            my_words[10]['text'] = r'$\eta$ = P(d|z, f) '

            # cols x
            c1, c2, c3 = 0.22, 0.5, 0.77
            h1, h2, h3 = 0.75, 0.24, 0.2

            my_words[0]['xy'] = (0.5, 0.85)
            my_words[1]['xy'] = (0.5, 0.93)

            my_words[2]['xy'] = (c1, h1)
            my_words[3]['xy'] = (c2, h1)
            my_words[4]['xy'] = (c3, h1)

            my_words[5]['xy'] = (c1, h2)
            my_words[6]['xy'] = (c2, h2)
            my_words[7]['xy'] = (c3, h2)

            my_words[8]['xy'] = (c1, h3)
            my_words[9]['xy'] = (c2, h3)
            my_words[10]['xy'] = (c3, h3)

        return my_words




