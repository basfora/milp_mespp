from milp_mespp.core import extract_info as ext
import numpy as np


class MySearcher:
    """Information related to each searcher, usually saved in a searchers[s] dict for s=1...m"""

    def __init__(self, searcher_id: int, v0: int, g, capture_range=0, zeta=None, my_seed=None):
        """ Init searcher class
        :param searcher_id : s = 1, 2...m
        :param v0 : start vertex v = 1,...,n
        :param g : graph
        :param capture_range : 0, 1, 2... (default is 0)
        :param zeta : 0 <= zeta < 1 (default is 0)
        :param my_seed"""

        # INITIAL -- immutable
        # ID for searcher instance, from 1...m
        self.id = searcher_id

        # CAPTURE INFORMATION
        # capture_range
        self.capture_range = capture_range
        # check for false negatives
        if zeta is not None:
            false_neg = True
        else:
            false_neg = False
        self.false_negative = false_neg
        self.zeta = zeta

        # get capture matrices
        self.capture_matrices = self.define_capture_matrix(g)

        # start vertex
        start = v0
        self.start = start

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

        # for sim repeatability
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
        """save the path planned by the milp algorithm for that searcher"""

        # get current time and vertex
        current_time, current_vertex = ext.get_last_info(self.path_taken)

        self.path_planned[current_time] = new_path_planned
        self.init_milp = new_path_planned[0]

    def get_path_planned(self, t=None):
        """Return the path planned as dict
        pi(s, t) = v"""

        pi = {}

        if t is None:
            t = ext.get_last_key(self.path_planned)

        list_v = self.path_planned[t]
        t_plan = 0
        for v in list_v:

            pi[(self.id, t_plan)] = v
            t_plan += 1

        return pi

    def get_last_planned(self):
        """Return the path planned as list of vertices"""

        t, list_v = ext.get_last_info(self.path_planned)

        return list_v

    def define_capture_matrix(self, g):

        V, n = ext.get_set_vertices(g)

        # size of capture matrix
        nu = n + 1

        my_aux = {}
        for v in V:
            # loop through vertices to get capture matrices
            C = self.rule_intercept(v, nu, self.capture_range, self.zeta, g)
            my_aux[v] = C
        C_all = my_aux

        return C_all

    def get_capture_matrix(self, u):
        """get capture matrices for a single searcher s and vertex u"""
        c_matrices = self.capture_matrices
        C = c_matrices.get(u)

        return C

    def get_all_capture_matrices(self):
        """get capture matrices for a single searcher s"""
        return self.capture_matrices

    @staticmethod
    def rule_intercept(v, nu, capture_range=0, zeta=None, g=None):
        """create C matrix based on the rule of interception
        graph needs to be an input if it's multiple vertices"""

        if zeta is None:
            my_type = 'int32'
            zeta = 0
        else:
            my_type = 'float64'

        # create identity matrix (n+1) x (n+1)
        C = np.identity(nu, my_type)

        # apply rule of interception to create capture matrices
        # same vertex (for now the same for each searcher, but can be different)
        # if integer, zeta = 0 and 1-zeta = 1
        if capture_range == 0:
            C[v][v] = zeta
            C[v][0] = 1 - zeta
        else:
            # list with all the vertices on the graph
            my_row = list(range(1, nu))
            for u in my_row:
                distance = ext.get_node_distance(g, v, u)
                if distance <= capture_range:
                    # assemble the capture matrix
                    C[u][u] = zeta
                    C[u][0] = 1 - zeta
        return C




















