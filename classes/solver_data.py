from igraph import *


class MySolverData:
    """Save at each time the MILP solver is called:
    - initial belief
    - path for the searchers
    - type of solution
    - objective function value
    - time needed to solve the model
    - gap
    """

    def __init__(self, horizon, deadline, theta, my_graph, solver_type='central'):
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
        self.g = Graph()
        self.g = my_graph

        self.theta = theta
        self.deadline = deadline

        self.solver_type = solver_type

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

        self.belief[t] = b_target

        # for different horizons (if necessary in the future)
        if horizon is None:
            horizon = self.horizon[0]
        self.horizon[t] = horizon
