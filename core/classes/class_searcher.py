import core.construct_model
import core.create_parameters
from core import extract_info as ext
from core import construct_model as cm


class MySearcher:
    """save information related to each searcher, for s=1...m"""

    def __init__(self, searcher_id: int, searchers_info: dict, capture_range=0, m=None, my_seed=None):

        # INITIAL (sim) -- immutable

        # ID for searcher instance, from 1...m
        self.id = searcher_id

        # repeat for all searchers
        if m is None:
            m = ext.get_set_searchers(searchers_info)[1]
        self.m = m

        # CAPTURE INFORMATION
        # capture_range
        self.capture_range = capture_range

        # get info out of searchers_info with that ID
        start = searchers_info[self.id]['start']
        self.start = start

        # get capture matrices
        self.capture_matrices = cm.get_all_capture_matrices(searchers_info, self.id)

        # check for false negatives
        false_neg, zeta = core.construct_model.check_false_negatives(searchers_info)
        self.false_negative = false_neg
        self.zeta = zeta

        # STORAGE (will be updated)
        # store planned path for each MILP (size = H) -- update each solver run
        self.path_planned = {}
        self.init_milp = start

        # store taken path for each time step (sim iteration)  -- update each time step
        self.path_taken = dict()
        self.path_taken[0] = start

        # keep track of current position
        self.current_pos = start

        # capture status
        self.catcher = False

        self.seed = my_seed

    def update_status(self, status=False):
        """True if this searchers captured the target"""
        self.catcher = status

    def evolve_position(self, next_position: int):
        """Evolve position of the searcher at each simulation iteration"""

        # get current time and vertex
        current_time, current_vertex = ext.get_last_info(self.path_taken)
        # next time step
        next_time = current_time + 1

        self.path_taken[next_time] = next_position

        self.current_pos = next_position

    def store_path_planned(self, new_path_planned: list):
        """save the path planned by the milp algorithm"""

        # get current time and vertex
        current_time, current_vertex = ext.get_last_info(self.path_taken)

        self.path_planned[current_time] = new_path_planned
        self.init_milp = new_path_planned[0]























