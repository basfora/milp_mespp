import datetime
from core import extract_info as ext


class MyInputs:

    def __init__(self):

        # graph
        self.graph = ext.get_graph_01()

        # solver inputs
        self.deadline = 10              # deadline (T)
        self.theta = 10                 # horizon (H)
        self.horizon = 10               # replan frequency (theta)
        self.gamma = 0.99               # cost function parameter
        self.solver_type = 'central'    # solver type, central or distributed

        # added 05-10
        self.timeout = 30*60
        self.iterations = 1

        # SEARCHER
        self.size_team = 1
        self.capture_range = 0
        self.zeta = None              # false negatives, zeta = 0.0 for no false negatives
        self.list_m = [1, 2, 3, 4, 5]   # maximum number of searchers, m = 1, 2, 3...
        self.searcher_seed_start = 1050

        # TARGET
        self.target_motion = 'random'   # random or static
        self.qty_possible_nodes = 5     # 5 possible vertices for starting position
        self.target_seed_start = 5000

        # BELIEF
        self.belief_distribution = 'uniform'

        # instance specific (special attention to these!)
        # belief: initial belief
        self.b0 = None
        self.v_taken = None

        # searcher initial position
        self.start_searcher_random = True
        self.start_searcher_v = None
        self.searcher_together = True

        # target initial position
        self.start_target_random = True
        self.start_target_true = None
        self.start_target_v_list = None

        # variables for iteration
        self.today_run = 1
        self.searcher_seed = 0
        self.target_seed = 0
        self.name_folder = ''

        # Monte Carlo Sim
        # first run of the day
        self.start_day = datetime.datetime.today().day
        # turns 0-runs_per_m
        self.runs_per_m = 20
        self.list_turns = list(range(self.today_run - 1, self.runs_per_m))

    def set_graph(self, graph_number_or_file):

        if isinstance(graph_number_or_file, int):
            if graph_number_or_file == 0:
                self.graph = ext.get_graph_00()
            elif graph_number_or_file == 1:
                # OFFICE
                self.graph = ext.get_graph_01()
            elif graph_number_or_file == 2:
                # GRID 10x10
                self.graph = ext.get_graph_02()
            elif graph_number_or_file == 3:
                # GRID 8x8
                self.graph = ext.get_graph_03()
            elif graph_number_or_file == 4:
                self.graph = ext.get_graph_04()
            elif graph_number_or_file == 5:
                self.graph = ext.get_graph_05()
            elif graph_number_or_file == 6:
                self.graph = ext.get_graph_06()
            elif graph_number_or_file == 7:
                # MUSEUM
                self.graph = ext.get_graph_07()
            elif graph_number_or_file == 8:
                # MUSEUM
                self.graph = ext.get_graph_08()
            else:
                print("No graph with that number")
        elif isinstance(graph_number_or_file, str):
            # if name of a file stored in graphs folder
            self.graph = ext.get_graph(graph_number_or_file)
        else:
            self.graph = graph_number_or_file

    def set_capture_range(self, value: int):
        self.capture_range = value

    def set_zeta(self, value: float):
        self.zeta = value

    def set_list_m(self, my_list: list):
        self.list_m = my_list

    def set_theta(self, value: int):
        self.theta = value

    def set_deadline(self, value: int):
        self.deadline = value

    def set_horizon(self, value: int):
        self.horizon = value

    def set_all_times(self, value: int):
        self.theta = value
        self.horizon = value
        self.deadline = value

    def set_solver_type(self, my_type: str):
        self.solver_type = my_type
        if my_type == 'distributed':
            self.timeout = 10

    def set_today_run(self, value: int):
        self.today_run = value

    def set_day(self, value: int):
        self.start_day = value

    def set_size_team(self, value: int):
        self.size_team = value

    def set_seeds(self, turn: int):
        self.searcher_seed = turn + self.searcher_seed_start
        self.target_seed = turn + self.target_seed_start

    def change_seed(self, my_seed: int, who: str):
        if who == 's':
            self.searcher_seed = my_seed
        elif who == 't':
            self.target_seed = my_seed

    def update_run_number(self):
        self.today_run = self.today_run + 1

    def set_number_of_runs(self, value):
        self.runs_per_m = value
        self.list_turns = list(range(self.today_run - 1, self.runs_per_m))

    def set_list_turns(self, min_range=1):
        self.list_turns = list(range(min_range - 1, self.runs_per_m))

    def set_qty_nodes(self, n_nodes):
        self.qty_possible_nodes = n_nodes

    def create_folder(self, folder_parent='data'):

        ext.folder_in_project(folder_parent)

        # create name with code
        name_folder, whole_path = ext.get_codename(self, folder_parent)

        # create new folder to save data
        ext.path_exists(whole_path)

        self.name_folder = name_folder
        return name_folder

    def set_start_seeds(self, searcher: int, target: int):
        self.searcher_seed_start = searcher
        self.target_seed_start = target

    def set_target_motion(self, my_motion: str):
        self.target_motion = my_motion

    def set_belief_distribution(self, type_of_belief: str):
        self.belief_distribution = type_of_belief

    def set_b0(self, b0):
        """include belief capture --> len(b0) = n + 1"""
        self.b0 = b0

    def set_timeout(self, timeout: int):
        self.timeout = timeout

    def set_start_searchers(self, v0: list):
        self.start_searcher_random = False
        self.start_searcher_v = v0
        self.size_team = len(v0)

    def set_start_target_true(self, v0: int):
        self.start_target_random = False
        self.start_target_true = v0

    def set_start_target(self, v_list: list):
        """Target will start at the first vertex of the list"""
        self.start_target_random = False
        self.start_target_v_list = v_list

        self.qty_possible_nodes = len(v_list)
        self.start_target_true = v_list[0]

    def set_v_taken(self, v_list: list, who: str):
        self.v_taken = v_list
        if who == 's':
            self.set_start_searchers(v_list)
        else:
            self.set_start_target(v_list)

    def set_searcher_together(self, op: bool):
        self.searcher_together = op






