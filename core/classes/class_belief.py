
from core import extract_info as ext
from core import aux_classes as ac


class MyBelief:
    """Define belief class
    Properties: initial belief (user input)
    belief at each time-step
    belief at time of re-planning
    """

    def __init__(self, sim_start_belief: list):
        """create instance of class with input belief"""
        # INITIAL
        self.start_belief = sim_start_belief

        # STORAGE
        self.stored = {}
        self.stored = {0: self.start_belief}

        # RECURSIVE
        # initial vertex recursive, for planning update, this is the initial belief for each re-plan (t%theta=0)
        # first re-plan is with initial belief provided
        self.milp_init_belief = self.start_belief
        self.new = sim_start_belief

    # change searchers info input after making searchers class (searchers position needs to be (s) = v
    def update(self, searchers_info: dict, pi_next_t: dict, Mtarget: list, n: int):
        """Use Eq. (2) to update current belief
        store belief vector based on new belief"""

        # get last time and belief
        current_time, current_belief = ext.get_last_info(self.stored)
        next_time = current_time + 1

        # find the product of the capture matrices, s = 1...m
        prod_C = ac.product_capture_matrix(searchers_info, pi_next_t, n)

        # assemble the matrix for multiplication
        big_M = ac.assemble_big_matrix(n, Mtarget)

        new_belief = ac.belief_update_equation(current_belief, big_M, prod_C)

        self.stored[next_time] = new_belief
        self.new = new_belief

    def new_init_belief(self, new_init_belief: list):
        """update recursive initial belief"""
        self.milp_init_belief = new_init_belief




