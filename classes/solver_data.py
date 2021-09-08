from core import extract_info as ext


class MySolverData:
    """Save at each time the MILP solver is called:
    - initial belief
    - path for the searchers
    - type of solution
    - objective function value
    - time needed to solve the model
    - gap
    """

    def __init__(self, horizon, deadline, theta, my_graph, solver_type='central', timeout=30*60):
        """ initialize after solver is first called
        :param number_searchers:
        """

        # objective function value for each time the solver runs
        self.obj_value = {}
        # solving time for each time the solver runs
        self.solve_time = {}
        # gap for each time the solver runs
        self.gap = {}

        self.x_s = {}

        self.belief = {}

        # horizon can be mutable in future experiments
        self.horizon = dict()
        self.horizon[0] = horizon

        # input parameters (immutable)
        self.g = None
        self.g = my_graph
        self.V, self.n = ext.get_set_vertices(my_graph)

        self.theta = theta
        self.deadline = deadline

        self.solver_type = solver_type
        self.timeout = timeout

        self.threads = {}

        self.gamma = 0.99

    def store_new_data(self, obj_fun, time_sol, gap, threads: int, x_s: dict, b_target: dict, t: int,
                       horizon=None):
        """call after solving each time
        s_pos[s] = {(v, t) = 0 or 1}"""

        self.obj_value[t] = obj_fun

        self.solve_time[t] = time_sol

        self.gap[t] = gap

        self.threads[t] = threads

        self.x_s[t] = x_s

        # b_target[(v, t)] = beta, 0 <= beta <= 1
        self.belief[t] = b_target

        # for different horizons (if necessary in the future)
        if horizon is None:
            horizon = self.horizon[0]
        self.horizon[t] = horizon

    # retrieving data
    def unpack(self):
        deadline = self.deadline
        horizon = self.horizon[0]
        theta = self.theta
        solver_type = self.solver_type
        gamma = self.gamma

        return deadline, horizon, theta, solver_type, gamma

    def unpack_for_planner(self):

        horizon = self.horizon[0]
        solver_type = self.solver_type
        gamma = self.gamma
        timeout = self.timeout
        g = self.g

        return g, horizon, solver_type, timeout, gamma

    def unpack_for_sim(self):

        deadline = self.deadline
        theta = self.theta
        n = self.n

        return deadline, theta, n

    def retrieve_graph(self):
        """Return stored graph and printer friendly layout"""

        g = self.g

        if 'grid' in g['name']:
            my_layout = g.layout("grid")
        else:
            my_layout = g.layout("kk")

        return g, my_layout

    def retrieve_solver_belief(self, t_plan=0, t=0):
        """Retrieve belief and return as list for desired time t
        :param t_plan : time the solver was called
        :param t : time you want the computed belief"""

        # get raw info stored from the solver
        # b_target[(v, t)] = beta, 0 <= beta <= 1
        b_target = self.belief[t_plan]

        # make it pretty: b = [b_c, b_v1, .... b_vn]
        belief = self.get_belief_vector(b_target, t)

        return belief

    def retrieve_planned_path(self, t_plan=0, s=None):
        """Retrieve planned path for searchers:
        t_plan : planned time
        if s = None, retrieve for all searchers
        return path as list
        path[s] = [v0, ...vh]"""

        x_s = self.x_s[t_plan]
        path = ext.xs_to_path_list(x_s)

        if s is not None:
            pi = path[s]
        else:
            pi = path

        return pi

    def retrieve_occupied_vertices(self, t_plan=0, t=0):
        """Retrieve list of occupied vertices at time t by searchers"""

        pi = self.retrieve_planned_path(t_plan)

        pi_t = []
        for s in pi.keys():
            v = pi[s][t]
            pi_t.append(v)

        return pi_t

    @staticmethod
    def get_belief_vector(b: dict, t: int):
        """Return belief vector
        b(t) = [b_c, b_v1, b_v2...b_vn]"""

        my_list = [k[1] for k in b.keys() if k[1] == t]

        # number of vertices + capture
        nu = len(my_list)
        # set of capture + vertices V_c = [0, 1, ... n]
        V_c = ext.get_idx_vertices(nu)[0]

        belief = []
        for v in V_c:
            beta = b.get((v, t))
            belief.append(beta)

        return belief
















