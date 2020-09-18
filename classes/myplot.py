import os


class PlotMyWay:

    def __init__(self, plot_op='box'):

        # these don't change
        # path to where the data is
        self.log_path = '../plot_data/'

        # number of run instances
        self.n_runs = 100
        self.list_runs = list(range(1, self.n_runs + 1))

        # overall deadline
        self.tau = 50

        # pick from these
        #
        self.all_scenarios = ['G1TNSV', 'G2TNMV', 'G2FNMV', 'G3TNSV']

        self.all_solver_types = ['C', 'D']

        self.all_x_vars = ['S', 'H']

        # to facilitate
        self.h_list = [6, 8, 10, 12, 14, 16, 18, 20]

        self.m_list = [1, 2, 3, 4, 5]

        self.m = self.m_list[0]
        self.h = self.h_list[0]

        # set initially
        # all scenarios
        self.scenario = self.all_scenarios
        # centralized
        self.solver = self.all_solver_types[0]
        # varying s
        self.key_l = self.all_x_vars[0]

        # specific for the plot you are choosing
        self.plot_type = plot_op

        self.my_list = self.m_list
        self.my_fixed = self.h

        self.x_label = ''

        self.base = {0: [], 1: []}
        self.parent = []
        self.n_plots = len(self.all_scenarios)

        self.complete_path = {}
        self.subfolders = {}

        self.fig_name = ""
        self.name_scenes = ''

        self.together = False
        self.difference = False
        self.code = None

        self.fit = False
        self.cpp = False

    def add_to_path(self, subfolder):
        self.log_path = self.log_path + subfolder + '/'

    def make_fit(self):
        self.fit = True

    def make_cpp(self):
        self.cpp = True

    def set_plot_type(self, op: str):
        """1: box plot
        2: objective function and mission time"""
        self.plot_type = op

    def set_solver_type(self, letter: str):
        """C, D or CD"""
        self.solver = letter

    def set_n_runs(self, value):
        self.n_runs = value
        self.list_runs = list(range(1, self.n_runs + 1))

    def set_scenario(self, str_list):

        scenes = []

        for el in str_list:
            real_name = self.code_scene(el)
            scenes.append(real_name)

        self.scenario = scenes
        self.name_scenes = str_list

        self.n_plots = len(scenes)

    def set_difference(self, code_list=None):

        code = []

        code[0:2] = code_list[0:2]

        if code_list is not None:

            for scene in code_list[2:]:
                code.append(self.code_scene(scene))

            self.code = code
            self.difference = True

    def set_together(self, together=False):
        self.together = together

    def code_scene(self, my_name):

        if my_name == 'MUSEUM':
            code_name = 'G1TNSV'

        elif my_name == 'GRID-NOFN':
            code_name = 'G2TNMV'

        elif my_name == 'GRID-FN':
            code_name = 'G2FNMV'

        elif my_name == 'OFFICE':
            code_name = 'G3TNSV'
        else:
            code_name = None

        return code_name

    def decode_scene(self, my_name):

        if my_name == 'G1TNSV':
            decode_name = 'MUSEUM'
        elif my_name == 'G2TNMV':
            decode_name = 'GRID-NOFN'
        elif my_name == 'G2FNMV':
            decode_name = 'GRID-FN'
        elif my_name == 'G3TNSV' :
            decode_name = 'OFFICE'
        else:
            decode_name = None

        return decode_name

    def set_key_l(self, my_l: str, value: int, other_list=None):
        """Set which is varying H/S and the value of fixed one (the other)"""

        self.key_l = my_l
        self.my_fixed = value

        if self.key_l == 'H':
            self.my_list = self.h_list
            self.m = self.my_fixed
        elif self.key_l == 'S' or 'T':
            self.my_list = self.m_list
            self.h = self.my_fixed
        else:
            pass

        if other_list is not None:
            self.my_list = other_list

    def set_fixed_var(self, value):
        self.my_fixed = value

    def set_m(self, value):
        self.m = value

    def set_h(self, value):
        self.h = value

    def get_folder_parent(self):

        ch = self.solver + 'H'
        # parent name will follow notation

        # put together name of folder (start of name)
        if self.key_l == 'S':
            # varying m
            S1 = str(self.my_fixed)
            S2 = str(self.my_list[0]) + '-' + str(self.my_list[-1])

            pre_base = ch + S1 + self.key_l
            pos_base = ''

        elif self.key_l == 'H':
            # varying h
            S1 = str(self.my_list[0]) + '-' + str(self.my_list[-1])
            S2 = str(self.my_fixed)

            pre_base = ch
            pos_base = 'S' + S2

        elif self.key_l == 'T':
            # varying m
            S1 = str(self.my_fixed) + 'T' + str(self.tau)
            S2 = str(self.my_list[0]) + '-' + str(self.my_list[-1])

            pre_base = ch + S1 + 'S'
            pos_base = ''
        else:
            S1, S2, pre_base, pos_base = '', '', '', ''

        # assemble start
        aux_name = ch + S1 + 'S' + S2

        parent = []
        base = {0: [], 1: []}
        for el in self.scenario:
            parent.append(aux_name + el)

            base[0].append(pre_base)
            base[1].append(pos_base + el)

        if self.cpp:
            milp_parent = parent
            new_parent = [m_f for m_f in parent]
            for folder in milp_parent:
                cpp_folder = folder + '_cpp'
                new_parent.append(cpp_folder)

            parent = new_parent

        self.parent = parent
        self.base = base

        return parent, base

    def get_folder_path(self):

        folder_interest = {}
        subfolders = {}
        for folder in self.parent:
            # get path for the parent folder and subfolders in it
            folder_interest[folder] = self.log_path + folder

            # sub folders of my folder of interest
            subfolders[folder] = os.listdir(folder_interest[folder])

        self.complete_path = folder_interest
        self.subfolders = subfolders

        return folder_interest, subfolders

    def get_label(self):

        if self.key_l == 'H':
            x_label = '$h$'
        else:
            x_label = '$m$'

        self.x_label = x_label

        return x_label

    def get_prepped(self):

        if len(self.solver) > 1:
            self.get_prepped_cd()
        elif self.key_l == 'T':
            self.get_prepped_tau()
        else:
            self.get_prepped_single()

    def get_prepped_single(self):
        self.get_folder_parent()
        self.get_folder_path()
        self.get_label()

    def get_prepped_cd(self):

        solver = ['C', 'D']

        temp_parent = []
        temp_base = {0: [], 1: []}

        for i in range(0, 2):

            self.set_solver_type(solver[i])

            aux_parent, aux_base = self.get_folder_parent()

            for el in aux_parent:
                idx = aux_parent.index(el)
                temp_parent.append(el)
                temp_base[0].append(aux_base[0][idx])
                temp_base[1].append(aux_base[1][idx])

        self.solver = 'CD'
        self.parent = temp_parent
        self.base = temp_base

        self.get_folder_path()
        self.get_label()

    def get_prepped_tau(self):

        h = [5, 10]

        temp_parent = []
        temp_base = {0: [], 1: []}

        for i in range(0, 2):

            self.my_fixed = h[i]

            aux_parent, aux_base = self.get_folder_parent()

            for el in aux_parent:
                idx = aux_parent.index(el)
                temp_parent.append(el)
                temp_base[0].append(aux_base[0][idx])
                temp_base[1].append(aux_base[1][idx])

        self.my_fixed = 10
        self.parent = temp_parent
        self.base = temp_base

        self.get_folder_path()
        self.get_label()

























