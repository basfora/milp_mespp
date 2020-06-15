import datetime
from core import extract_info as ext
import os


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
        self.searcher_seed = 0
        self.searcher_seed_start = 1050

        # TARGET
        self.target_motion = 'random'   # random or static
        self.qty_possible_nodes = 5     # 5 possible vertices for starting position
        self.target_seed = 0
        self.target_seed_start = 5000

        # BELIEF
        self.belief_distribution = 'uniform'

        # variables for iteration
        self.start_day = datetime.datetime.today().day

        self.today_run = 1
        # turns 0-runs_per_m
        self.runs_per_m = 20
        self.list_turns = list(range(self.today_run-1, self.runs_per_m))

        self.name_folder = ''

    def get_graph(self, graph_number: int):

        if graph_number == 1:
            # MUSEUM
            self.graph = ext.get_graph_01()
        elif graph_number == 2:
            # GRID 10x10
            self.graph = ext.get_graph_02()
        elif graph_number == 3:
            # GRID 8x8
            self.graph = ext.get_graph_03()
        elif graph_number == 4:
            self.graph = ext.get_graph_04()
        elif graph_number == 5:
            self.graph = ext.get_graph_05()
        elif graph_number == 6:
            self.graph = ext.get_graph_06()
        elif graph_number == 7:
            # OFFICE
            self.graph = ext.get_graph_07()
        else:
            print("No graph with that number")

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

        # create name with code
        name_folder, whole_path = ext.get_codename(self, folder_parent)

        # print(whole_path)
        # create new folder to save figures
        if not os.path.exists(whole_path):
            os.mkdir(whole_path)
        else:
            print("Directory " + name_folder + " already exists")

        self.name_folder = name_folder
        return name_folder

    def set_start_seeds(self, searcher: int, target: int):
        self.searcher_seed_start = searcher
        self.target_seed_start = target

    def set_target_motion(self, my_motion: str):
        self.target_motion = my_motion

    def set_belief_distribution(self, b0):
        self.belief_distribution = b0

    def set_timeout(self, timeout: int):
        self.timeout = timeout




