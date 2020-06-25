from core import extract_info as ext
from core import plot_fun as pf
import pickle
from igraph import *


class MyHazard:
    """Define class of Hazard
    Class is defined with variables for smoke, but its the same for fire!
    Properties:
    type: fire (f), smoke (z)

    """

    def __init__(self, g, deadline, what='smoke', h0=None):

        # fire or smoke
        self.type = what
        self.g_name = ""
        self.g = None
        # name of folder for this sim: eg smoke_G9V_grid_date#_run#
        self.folder_name = ""
        # whole path + /name_folder
        self.whole_path = ""

        if h0 is None:
            h0 = 1

        # ---------------------
        # pre defined parameters
        # ---------------------
        # hazard levels
        self.levels = [1, 2, 3, 4, 5]
        self.n_levels = len(self.levels)
        self.level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        self.level_color = ['green', 'blue', 'yellow', 'orange', 'red']
        # smoke (z) -- DEFAULT
        self.default_z_mu = 0.3
        self.default_z_sigma = 0.1
        # fire (f)
        self.default_f_mu = 0.2
        self.default_f_sigma = 0.05
        # spread (both)
        self.lbda_mu = 0.02
        self.lbda_sigma = 0.005
        # -----------------------

        # vertices
        self.n = 0
        self.V = []

        # connectivity
        self.E = []

        # time related info
        self.tau = 0
        self.T = []
        self.T_ext = [0]

        # initial hazard value
        self.z_0 = {}

        # values throughout time
        self.z = dict()
        self.z_level = dict()

        self.z_only = dict()
        self.z_joint = dict()

        self.create_empty_dicts()

        # parameters
        # linear coefficient -- smoke or fire
        # do not vary with time
        self.xi = dict()
        self.xi_mu = 0.0
        self.xi_sigma = 0.0

        # spread coefficient (lambda)
        self.lbda = dict()

        # call initializing functions
        self.structure(g, deadline)
        self.init_default(h0)

    # --------------
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

        # get connectivity matrix (list form)
        self.E = ext.get_connectivity_matrix(g)

        # init 0 populated or empty dictionaries: z, xi, lbda
        self.create_empty_dicts()

    def create_empty_dicts(self):
        """Init dictionary to evolve each time"""

        for v in self.V:

            # hazard -- float
            # z(v, t)
            self.z[v] = {}

            # z_only(v,t)
            self.z_only[v] = {}

            # z_spread(v,t)
            self.z_joint[v] = {}
            self.z_joint[v][0] = 0

            # hazard level -- int
            # z_level(v,t)
            self.z_level[v] = {}

            # xi(v)
            self.xi[v] = None

            # spread lambda(v,u)
            self.lbda[v] = {}
            for u in self.V:
                self.lbda[v][u] = None

    def init_default(self, h0):
        """set parameters according to default values"""

        self.xi_default()
        # call these with default values
        self.set_param_vertices()
        self.init_hazard(h0)

    def xi_default(self):
        """Assign defaults iota (fire) or xi (smoke) to linear coefficient """

        if self.type == 'fire':
            self.xi_mu = self.default_f_mu
            self.xi_sigma = self.default_f_sigma

        elif self.type == 'smoke':
            self.xi_mu = self.default_z_mu
            self.xi_sigma = self.default_z_sigma

        else:
            print('No other hazard.')

    def set_param_vertices(self):
        """Set evolution parameters for each v, u
        Call this after setting xi_mu, xi_sigma , lbda_mu, lbda_sigma"""

        # number of vertices
        n = self.n

        my_samples = ext.get_sample_normal(n, self.xi_mu, self.xi_sigma)

        for v in self.V:
            v_idx = ext.get_python_idx(v)

            # coefficient of cell evolution
            self.xi[v] = my_samples[v_idx]

            # coefficient of spread
            for u in self.V:
                # check to see if its not filled already
                if self.lbda[v][u] is None:
                    my_lambda = ext.get_sample_normal(1, self.lbda_mu, self.lbda_sigma)
                    self.lbda[v][u] = my_lambda[0]
                    self.lbda[u][v] = my_lambda[0]

    # --------------------
    # user input values
    def change_param(self, mu: float, sigma: float, name='xi'):
        """Change mu and sigma for random linear coefficients
        name = xi or lambda"""

        if name is not 'xi':
            # change spread
            self.lbda_mu = mu
            self.lbda_sigma = sigma

        else:
            # change xi
            self.xi_mu = mu
            self.xi_sigma = sigma

        # init 0 populated or empty dictionaries: z, xi, lbda
        self.create_empty_dicts()

        # update for all vertices
        self.set_param_vertices()

    def init_hazard(self, h0):
        """Assign level value for each v at t = 0"""

        for v in self.V:
            # h is an integer
            if isinstance(h0, int):
                h = h0
            # h is a list with values for each vertex
            else:
                v_idx = ext.get_python_idx(v)
                h = h0[v_idx]

            self.set_v_value(v, h, 0)

    def set_v_value(self, v: int, value_raw: int, t=0):
        """define level for specific vertex v = [1, ...n]"""

        value = self.saturate_value(value_raw)

        if t == 0:
            self.z_0[v] = value
            self.z_only[v][t] = value
            self.z_level[v][t] = value

        self.z[v][t] = value

    # -------------------
    # evolution stuff
    def evolve(self, t=None):

        # set time if not given
        if t is None:
            t = ext.get_last_key(self.z[1]) + 1

        # loop through vertices
        for v in self.V:
            # get isolated hazard evolution
            self.update_isolated(v, t)
            # get contribution from adjacent vertices
            self.update_joint(v, t)

            # combine both
            z = self.z_only[v][t] + self.z_joint[v][t]

            # update value in class
            self.z[v][t] = z

            # round and saturate
            self.update_level(v, t)

    def update_isolated(self, v: int, t: int):
        """evolve isolated cell
           z(t) = xi*t + z(0)"""

        z = self.xi[v] * t + self.z[v][0]

        self.z_only[v][t] = z

        return z

    def update_joint(self, v: int, t: int):
        """evolve joint hazard
        Sum [e_vu * lambda_vu * z_u(t-1)"""

        my_z = 0
        for u in self.V:
            # check if there an edge there
            e_vu = ext.has_edge(self.E, v, u)
            lbd_vu = self.lbda[v][u]
            z_u = self.z[u][t-1]

            z_vu = e_vu * lbd_vu * z_u

            my_z = my_z + z_vu

            self.z_joint[v][t] = my_z

        return my_z

        # extract info

    def update_level(self, v: int, t: int):
        """round z(v,t) to nearest integer
        saturate to 1-5"""

        x = self.z[v][t]

        z = self.get_level(x)

        self.z_level[v][t] = z

        return z

    # ------------------
    # extract info (doesn't change anything on class, just retrieve info)
    def get_level(self, x: float):
        """round to nearest integer
        saturate to 1-5"""

        x_int = round(x)
        z_level = self.saturate_value(x_int)

        return z_level

    def saturate_value(self, z):
        max_level = self.levels[-1]

        if z > max_level:
            z = max_level
        return z

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

    def get_value_vt(self, v: int, t: int, op=0):
        """Retrieve z(v,t) : level, total, only or from spread"""

        if op == 0:
            z = self.z_level[v][t]
        elif op == 1:
            z = self.z[v][t]
        elif op == 2:
            z = self.z_only[v][t]
        else:
            z = self.z_joint[v][t]

        return z

    def get_value_t(self, t: int, op=0):
        """Retrieve z at time t: level, total, only or from spread"""

        z = {}
        for v in self.V:
            if op == 0:
                my_z = self.z_level[v][t]
            elif op == 1:
                my_z = self.z[v][t]
            elif op == 2:
                my_z = self.z_only[v][t]
            else:
                my_z = self.z_joint[v][t]
            z[v] = my_z

        # print(z)
        return z

    def get_value_v(self, v: int, op=0):
        """Retrieve z for vertex v: level, total, only or from spread"""

        z = {}
        for t in self.T:
            if op == 0:
                my_z = self.z_level[v][t]
            elif op == 1:
                my_z = self.z[v][t]
            elif op == 2:
                my_z = self.z_only[v][t]
            else:
                my_z = self.z_joint[v][t]
            z[t] = my_z

        print(z)
        return z

    def get_value_all(self, op=0):
        """Retrieve z for all v,t: level, total, only or from spread"""

        if op == 0:
            z = self.z_level
        elif op == 1:
            z = self.z
        elif op == 2:
            z = self.z_only
        else:
            z = self.z_joint

        print(z)
        return z

    # --------------------
    # simulate
    def simulate(self, op=2):
        """simulate the hazard evolution and plot results"""

        # simulate
        for t in self.T:
            self.evolve(t)

        if op == 0:
            self.save_data()
        if op == 1:
            self.plot_simple()
        elif op == 2:
            self.plot_frames()
        elif op == 3:
            self.make_video()
        else:
            return self

    # -----------------------
    # save in pickle file

    def save_data(self):

        # name the pickle file
        self.create_folder()
        file_name = 'data_save.txt'
        full_path = self.whole_path + "/" + file_name

        my_pickle = open(full_path, "wb")
        pickle.dump(self, my_pickle)
        my_pickle.close()

        print("Data saved in: ", self.folder_name)
        return

    def create_folder(self, i=1):

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

    # plot evolution
    def plot_simple(self):

        # save data first
        self.save_data()

        for t in self.T_ext:
            z_t = self.get_value_t(t)
            self.plot_graph(z_t, t)

        return

    def plot_graph(self, z_t: dict, t: int):
        """z_t: dict z[v] v= 1,...n"""

        # get graph layout
        g = self.g
        my_layout = g.layout("grid")

        folder_path = self.whole_path

        for v in self.V:
            v_idx = ext.get_python_idx(v)
            my_color = self.get_level_color(z_t[v])
            # assign color
            g.vs[v_idx]["color"] = my_color

        name_file = folder_path + "/" + self.type + "_t" + str(t) + ".png"
        # name_fig = self.type + "_t" + str(t) + ".png"
        plot(g, name_file, layout=my_layout, figsize=(3, 3), bbox=(400, 400), margin=15, dpi=400)

        return name_file

    def frame_details(self):

        if self.type == 'smoke':
            var = r'$\xi$'
            name_hazard = 'Smoke'
        else:
            var = r'$\iota$'
            name_hazard = 'Fire'

        my_words = pf.empty_my_words(2)

        # title , time step, subtitle
        my_words[0]['text'] = name_hazard + ' Evolution, t = '

        my_words[1]['text'] = var + '~' + "(" + str(self.xi_mu) + ',' + str(self.xi_sigma) + r'$^2$' + ')' + ', ' + r'$\lambda$ ~ (' + str(self.lbda_mu) + ',' + str(self.lbda_sigma) + r'$^2$' + ')'

        my_words[0]['xy'] = (0.5, 0.93)
        my_words[1]['xy'] = (0.5, 0.88)
        # my_words[2]['xy'] = (0.5, 0.93)

        return my_words

    def plot_frames(self):

        # save data first
        self.save_data()

        for t in self.T_ext:
            z_t = self.get_value_t(t)
            hazard_t = self.plot_graph(z_t, t)
            my_words = self.frame_details()
            # file name, time step and labels
            pf.mount_frame(hazard_t, t, my_words)

        pf.delete_frames(self.whole_path, self.type)

        return

    def make_video(self):
        # save data first
        self.save_data()

        for t in self.T_ext:
            z_t = self.get_value_t(t)
            hazard_t = self.plot_graph(z_t, t)
            my_words = self.frame_details()
            # file name, time step and labels
            pf.mount_frame(hazard_t, t, my_words, 1, True)

        # compose short video
        pf.compose_and_clean(self.whole_path)






















