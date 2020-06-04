import numpy as np
from core import extract_info as ext
from core import construct_model as cm


def assemble_big_matrix(n: int, Mtarget):
    """Assemble array for belief update equation"""

    if isinstance(Mtarget, list):
        # transform motion matrix in array
        M = np.asarray(Mtarget)
    else:
        M = Mtarget

    # assemble the array
    a = np.array([1])
    b = np.zeros((1, n), dtype=int)
    c = np.zeros((n, 1), dtype=int)
    # extended motion array
    row1 = np.concatenate((a, b), axis=None)
    row2 = np.concatenate((c, M), axis=1)
    # my matrix
    big_M = np.vstack((row1, row2))
    return big_M


def change_type(A, opt: str):
    """Change from list to array or from array to list"""
    # change to list
    if opt == 'list':
        if not isinstance(A, np.ndarray):
            B = False
        else:
            B = A.tolist()
    # change to array
    elif opt == 'array':
        if not isinstance(A, list):
            B = False
        else:
            B = np.asarray(A)
    else:
        print("Wrong type option, array or list only")
        B = False

    return B


def sample_vertex(my_vertices: list, prob_move: list):
    """ sample 1 vertex with probability weight according to prob_move"""
    # uncomment for random seed
    ext.get_random_seed()
    my_vertex = np.random.choice(my_vertices, None, p=prob_move)
    return my_vertex


def probability_move(M, current_vertex):
    """get moving probabilities for current vertex"""

    # get current vertex id
    v_idx = ext.get_python_idx(current_vertex)
    n = len(M[v_idx])
    prob_move = []
    my_vertices = []
    for col in range(0, n):
        prob_v = M[v_idx][col]
        # if target can move to this vertex, save to list
        if prob_v > 1e-4:
            prob_move.append(prob_v)
            my_vertices.append(col + 1)

    return my_vertices, prob_move


# change searchers info input after making searchers class (searchers position needs to be dict with key (s, t)
def product_capture_matrix(searchers_info: dict, pi_next_t: dict, n: int):
    """Find and multiply capture matrices for s = 1,...m"""

    # number of vertices + 1
    nu = n + 1
    C = {}
    prod_C = np.identity(nu)
    # get capture matrices for each searcher that will be at pi(t+1)
    for s in searchers_info.keys():
        # get where the searchers is now
        v = pi_next_t.get(s)
        # extract the capture matrix for that vertex
        C[s] = cm.get_capture_matrix(searchers_info, s, v)
        # recursive product of capture matrix, from 1...m searchers
        prod_C = np.matmul(prod_C, C[s])

    return prod_C


def belief_update_equation(current_belief: list, big_M: np.ndarray, prod_C: np.ndarray):
    """Update the belief based on Eq (2) of the model:
    b(t+1) = b(t) * big_M * Prod_C"""

    # transform into array for multiplication
    current_b = change_type(current_belief, 'array')

    # use belief update equation
    dummy_matrix = np.matmul(big_M, prod_C)
    new_b = np.matmul(current_b, dummy_matrix)

    # transform to list
    new_belief = change_type(new_b, 'list')

    return new_belief


def get_true_position(v_target, idx=None):
    """return true position of the target based on the initial vertice distribution
    idx is the index correspondent of the true position of the list v_target"""

    # if there is only one possible vertex, the target is there
    if len(v_target) == 1:
        v_target_true = v_target[0]
    else:
        # if no index for the true vertex was provided, choose randomly
        if idx is None:
            n_vertex = len(v_target)
            prob_uni = (1/n_vertex)
            my_array = np.ones(n_vertex)
            prob_array = prob_uni * my_array
            prob_move = prob_array.tolist()
            v_target_true = sample_vertex(v_target, prob_move)
        # if the index was provided, simply get the position
        else:
            if idx >= len(v_target):
                v_target_true = v_target[-1]
            else:
                v_target_true = v_target[idx]

    return v_target_true
