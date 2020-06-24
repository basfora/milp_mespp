from core import construct_model as cm
from core import extract_info as ext


class MyTarget:
    """Define class of target
    Properties:
    initial true location;

    location for each time step
    status capture
    """

    def __init__(self, plan_position: list, motion_matrix: list, true_position=None, my_seed=None, true_motion=None):
        """create instance of class with input position and belief"""

        # INITIAL
        # information fed to the planning (may be more than one vertex)
        self.start_possible = plan_position
        self.motion_matrix = motion_matrix
        self.seed = my_seed

        # if no true information was provided, use the information used in the planning
        # true position is always a single vertex
        if true_position is None:
            if len(plan_position) == 1:
                true_position = plan_position[0]
            else:
                print("Please provide a single start vertex for the target simulation.")
                exit()
        if true_motion is None:
            true_motion = motion_matrix

        # true information for simulation
        self.start_true = true_position
        self.motion_matrix_true = true_motion

        # STORAGE
        # properties for time-updates (planning)
        # create dict to store target position at each step
        self.stored_v_possible = {}
        # at time 0, target is in the initial position
        self.stored_v_possible = {0: self.start_possible}

        # properties for time-updates (simulation)
        self.stored_v_true = {}
        self.stored_v_true = {0: self.start_true}

        self.current_pos = self.start_true

        # target status of capture
        # only set to true once target is captured
        self.is_captured = False
        self.capture_v = None

        self.capture_time = None

        # RECURSIVE
        # initial vertex recursive, for planning update, this is the initial belief for each re-plan (t%theta=0)
        # first re-plan is with initial belief provided
        self.milp_init_v_possible = self.start_possible
        self.milp_init_v_true = self.start_true

    def update_status(self, status=False):

        time, vertex = ext.get_last_info(self.stored_v_true)
        self.is_captured = status
        self.capture_time = time
        self.capture_v = vertex
        # if status is True:
        #    print("Target was captured!")

    def evolve_true_position(self):
        """evolve the target position based on the probability given by the true motion matrix
        which vertex is now, sample with probability into neighboring vertices
        To be called each time step of the simulation"""

        # get motion matrix
        M = self.motion_matrix_true
        # get current time and vertex
        current_time, current_vertex = ext.get_last_info(self.stored_v_true)
        # next time step
        next_time = current_time + 1

        # get moving probabilities for current vertex
        my_vertices, prob_move = cm.probability_move(M, current_vertex)

        # sample 1 vertex with probability weight according to prob_move
        new_vertex = cm.sample_vertex(my_vertices, prob_move)

        # update true position
        self.stored_v_true[next_time] = new_vertex
        self.current_pos = new_vertex

    def evolve_possible_position(self, updated_belief: list):
        """Use belief vector to store which vertices the target might be in (probability > 0)"""

        # get current time and vertex
        current_time = ext.get_last_info(self.stored_v_possible)[0]
        # next time step
        next_time = current_time + 1

        new_vertices = []
        n = len(updated_belief)
        # get vertices where probability is higher than zero
        for i in range(1, n):
            if updated_belief[i] > 1e-4:
                new_vertices.append(i)

        # update the planning information
        self.stored_v_possible[next_time] = new_vertices

    def replace_init_position(self, new_v_possible: list, new_v_true=None):
        """save separately to be easier to retrieve for milp
        the initial position for that partilar planning instance with horizon H"""

        self.milp_init_v_possible = new_v_possible

        if new_v_true is None:
            if len(new_v_possible) == 1:
                new_v_true = new_v_possible[0]
            else:
                print("Please provide a single start vertex for the target simulation.")
                exit()

        self.milp_init_v_true = new_v_true





