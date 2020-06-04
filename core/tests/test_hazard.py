from core import extract_info as ext
from core.classes.class_hazard import MyHazard
import numpy as np


def get_param():

    g = ext.get_graph_04()
    deadline = 10

    # coefficient parameters
    default_z_mu = 0.3
    default_z_sigma = 0.1
    # fire (f)
    default_f_mu = 0.2
    default_f_sigma = 0.05
    # spread (both)
    lbda_mu = 0.02
    lbda_sigma = 0.005
    h0 = 1

    default_list = [default_z_mu, default_z_sigma, default_f_mu, default_f_sigma, lbda_mu, lbda_sigma, h0]

    return g, deadline, default_list


def test_init_default():
    """initialize my hazard -- smoke"""

    g, deadline, d_list = get_param()

    assert deadline == 10
    assert g["name"] == 'G9V_grid.p'

    hazard = MyHazard(g, deadline)

    # test pre defined parameters
    assert hazard.levels == [1, 2, 3, 4, 5]
    assert hazard.level_label == ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    assert hazard.level_color == ['green', 'blue', 'yellow', 'orange', 'red']
    assert hazard.n_levels == 5
    assert hazard.default_z_mu == d_list[0]
    assert hazard.default_z_sigma == d_list[1]
    assert hazard.default_f_mu == d_list[2]
    assert hazard.default_f_sigma == d_list[3]
    assert hazard.lbda_mu == d_list[4]
    assert hazard.lbda_sigma == d_list[5]

    # test graph-related parameters
    assert hazard.n == 9
    assert hazard.V == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert hazard.tau == deadline
    assert hazard.T == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # coefficients and type
    assert hazard.type == 'smoke'
    assert hazard.xi_mu == d_list[0]
    assert hazard.xi_sigma == d_list[1]

    # z values
    for v in hazard.V:
        t = 0
        assert hazard.z_0[v] == d_list[6]
        assert hazard.z[v][t] == d_list[6]
        assert hazard.z_level[v][t] == d_list[6]
        assert hazard.z_only[v][t] == d_list[6]
        assert hazard.z_joint[v][t] == 0


def test_connectivity():

    g, deadline, d_list = get_param()
    smoke = MyHazard(g, deadline)

    # test connectivity matrix
    # vertex 1
    i = ext.get_python_idx(1)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 1
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 1
    assert smoke.E[i][4] == 0
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 0

    # vertex 2
    i = ext.get_python_idx(2)
    assert smoke.E[i][0] == 1
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 1
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 1
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 0

    # vertex 3
    i = ext.get_python_idx(3)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 1
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 0
    assert smoke.E[i][5] == 1
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 0

    # vertex 4
    i = ext.get_python_idx(4)
    assert smoke.E[i][0] == 1
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 1
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 1
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 0

    # vertex 5
    i = ext.get_python_idx(5)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 1
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 1
    assert smoke.E[i][4] == 0
    assert smoke.E[i][5] == 1
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 1
    assert smoke.E[i][8] == 0

    # vertex 6
    i = ext.get_python_idx(6)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 1
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 1
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 1

    # vertex 7
    i = ext.get_python_idx(7)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 1
    assert smoke.E[i][4] == 0
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 1
    assert smoke.E[i][8] == 0

    # vertex 8
    i = ext.get_python_idx(8)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 1
    assert smoke.E[i][5] == 0
    assert smoke.E[i][6] == 1
    assert smoke.E[i][7] == 0
    assert smoke.E[i][8] == 1

    # vertex 9
    i = ext.get_python_idx(9)
    assert smoke.E[i][0] == 0
    assert smoke.E[i][1] == 0
    assert smoke.E[i][2] == 0
    assert smoke.E[i][3] == 0
    assert smoke.E[i][4] == 0
    assert smoke.E[i][5] == 1
    assert smoke.E[i][6] == 0
    assert smoke.E[i][7] == 1
    assert smoke.E[i][8] == 0


def test_fire():

    g, deadline, d_list = get_param()
    h0 = 2
    d_list[-1] = h0

    hazard = MyHazard(g, deadline, 'fire', h0)

    # test pre defined parameters
    assert hazard.levels == [1, 2, 3, 4, 5]
    assert hazard.level_label == ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    assert hazard.level_color == ['green', 'blue', 'yellow', 'orange', 'red']
    assert hazard.default_z_mu == d_list[0]
    assert hazard.default_z_sigma == d_list[1]
    assert hazard.default_f_mu == d_list[2]
    assert hazard.default_f_sigma == d_list[3]
    assert hazard.lbda_mu == d_list[4]
    assert hazard.lbda_sigma == d_list[5]

    # test graph-related parameters
    assert hazard.n == 9
    assert hazard.V == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert hazard.tau == deadline
    assert hazard.T == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # coefficients and type
    assert hazard.type == 'fire'
    assert hazard.xi_mu == d_list[2]
    assert hazard.xi_sigma == d_list[3]

    # z values
    for v in hazard.V:
        t = 0
        assert hazard.z_0[v] == d_list[6]
        assert hazard.z[v][t] == d_list[6]
        assert hazard.z_level[v][t] == d_list[6]
        assert hazard.z_only[v][t] == d_list[6]
        assert hazard.z_joint[v][t] == 0


def test_normal_distribution():

    g, deadline, d_list = get_param()

    hazard = MyHazard(g, deadline)

    mu = hazard.xi_mu
    sigma = hazard.xi_sigma

    mu2 = hazard.lbda_mu
    sigma2 = hazard.lbda_sigma

    assert mu == hazard.default_z_mu
    assert sigma == hazard.default_z_sigma

    my_samples = []
    my_samples2 = []

    for v in hazard.V:
        my_samples.append(hazard.xi[v])
        for u in hazard.V:
            my_samples2.append(hazard.lbda[v][u])

    my_mean = np.mean(my_samples)
    my_sigma = np.std(my_samples, ddof=1)

    my_mean2 = np.mean(my_samples2)
    my_sigma2 = np.std(my_samples2, ddof=1)

    assert round(abs(mu - my_mean), 2) <= 2*sigma
    assert round(abs(sigma - my_sigma), 2) <=2*sigma

    assert round(abs(mu2 - my_mean2), 2) <= 2*sigma2
    assert round(abs(sigma2 - my_sigma2), 2) <= 2*sigma2


def test_normal_distribution_fire():

    g, deadline, d_list = get_param()

    hazard = MyHazard(g, deadline, 'fire')

    mu = hazard.xi_mu
    sigma = hazard.xi_sigma

    mu2 = hazard.lbda_mu
    sigma2 = hazard.lbda_sigma

    assert mu == hazard.default_f_mu
    assert sigma == hazard.default_f_sigma

    my_samples = []
    my_samples2 = []

    for v in hazard.V:
        my_samples.append(hazard.xi[v])
        for u in hazard.V:
            my_samples2.append(hazard.lbda[v][u])

    my_mean = np.mean(my_samples)
    my_sigma = np.std(my_samples, ddof=1)

    my_mean2 = np.mean(my_samples2)
    my_sigma2 = np.std(my_samples2, ddof=1)

    assert abs(mu - my_mean) < 2*sigma
    assert abs(sigma - my_sigma) < 2*sigma

    assert abs(mu2 - my_mean2) < 2*sigma2
    assert abs(sigma2 - my_sigma2) < 2*sigma2


def test_change_parameters():

    g, deadline, d_list = get_param()

    hazard = MyHazard(g, deadline)

    # change xi
    mu_old = hazard.xi_mu
    sigma_old = hazard.xi_sigma
    mu = 0.5
    sigma = 0.05

    hazard.change_param(mu, sigma, 'xi')
    # make sure xi parameter was changed
    assert mu_old != hazard.xi_mu
    assert sigma_old != hazard.xi_sigma
    assert mu == hazard.xi_mu
    assert sigma == hazard.xi_sigma

    # change lambda
    mu2_old = hazard.lbda_mu
    sigma2_old = hazard.lbda_sigma

    mu2 = 0.3
    sigma2 = 0.05
    hazard.change_param(mu2, sigma2, 'lambda')

    # make sure lambda parameter was changed
    assert mu2_old != hazard.lbda_mu
    assert sigma2_old != hazard.lbda_sigma
    assert mu2 == hazard.lbda_mu
    assert sigma2 == hazard.lbda_sigma

    # collect samples in a list
    my_samples = []
    my_samples2 = []
    for v in hazard.V:
        my_samples.append(hazard.xi[v])
        for u in hazard.V:
            my_samples2.append(hazard.lbda[v][u])

    # compute xi mean and std
    my_mean = np.mean(my_samples)
    my_sigma = np.std(my_samples, ddof=1)
    # compute lambda mean and std
    my_mean2 = np.mean(my_samples2)
    my_sigma2 = np.std(my_samples2, ddof=1)

    # compare samples and truth
    # xi
    assert round(abs(mu - my_mean), 2) <= 2*sigma
    assert round(abs(sigma - my_sigma), 2) <= 2*sigma

    # lambda
    assert round(abs(mu2 - my_mean2), 2) <= 2*sigma2
    assert round(abs(sigma2 - my_sigma2), 2) <= 2*sigma2

    h_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    hazard.init_hazard(h_list)

    for v in hazard.V:
        if int(v) < 6:
            i = v -1
            j = h_list[i]
        else:
            j = 5
        assert hazard.z_0[v] == j


def test_update_isolated():
    g, deadline, d_list = get_param()

    # start with default values
    hazard = MyHazard(g, deadline)

    t = 1

    z_0 = hazard.z_0
    xi = hazard.xi

    z_only = {}
    for v in hazard.V:
        z_only[v] = {}
        z_only[v][t] = xi[v] * t + z_0[v]
        hazard.update_isolated(v, t)
        assert hazard.z_only[v][t] == z_only[v][t]
        assert isinstance(hazard.z_only[v][t], float)


def test_update_joint():
    g, deadline, d_list = get_param()
    # start with default values
    hazard = MyHazard(g, deadline)
    t = 1

    z_0 = hazard.z_0
    xi = hazard.xi
    my_lbda = hazard.lbda

    z_only = {}
    z_joint = {}

    for v in hazard.V:
        # start dict
        z_only[v] = {}
        z_joint[v] = {}

        # fill in
        z_only[v][t] = xi[v] * t + z_0[v]
        z_joint[v][t] = 0

    # connections
    u = {1: [2, 4], 2: [1, 5, 3], 3: [2, 6], 4: [1, 5, 7], 5: [2, 4, 6, 8], 6: [3, 5, 9], 7: [4, 8], 8: [5, 7, 9],
         9: [6, 8]}
    for v in hazard.V:
        # manual
        my_sum = 0
        for i in u[v]:
            assert isinstance(my_lbda[v][i], float)
            assert isinstance(hazard.lbda[v][i], float)
            assert isinstance(z_0[i], int)
            print(str(v) + '---' + str(i))
            assert ext.has_edge(hazard.E, v, i)

            my_sum = my_sum + (my_lbda[v][i] * z_0[i])
        # fill in
        z_joint[v][t] = my_sum

        # call function
        hazard.update_joint(v, t)

        # compare
        assert round(hazard.z_joint[v][t], 4) == round(z_joint[v][t], 4)
        assert isinstance(hazard.z_joint[v][t], float)


def test_evolve_hazard():
    g, deadline, d_list = get_param()
    # start with default values
    hazard = MyHazard(g, deadline)
    t = 1

    z_0 = hazard.z_0
    xi = hazard.xi
    my_lbda = hazard.lbda

    z_only = {}
    z_joint = {}
    z = {}
    z_level = {}
    for v in hazard.V:
        # start dict
        z_only[v] = {}
        z_joint[v] = {}
        z[v] = {}
        z_level[v] = {}

        # fill in
        z_only[v][t] = xi[v] * t + z_0[v]
        z_joint[v][t] = 0
        z[v][t] = 0
        z_level[v][t] = 0

    # call function
    hazard.evolve(t)

    # connections
    u = {1: [2, 4], 2: [1, 5, 3], 3: [2, 6], 4: [1, 5, 7], 5: [2, 4, 6, 8], 6: [3, 5, 9], 7: [4, 8], 8: [5, 7, 9],
         9: [6, 8]}
    for v in hazard.V:
        # manual
        my_sum = 0
        for i in u[v]:
            assert isinstance(my_lbda[v][i], float)
            assert isinstance(hazard.lbda[v][i], float)
            assert isinstance(z_0[i], int)
            print(str(v) + '---' + str(i))
            assert ext.has_edge(hazard.E, v, i)

            my_sum = my_sum + (my_lbda[v][i] * z_0[i])
        # fill in
        z_joint[v][t] = my_sum
        z[v][t] = z_only[v][t] + z_joint[v][t]

        z_aux = round(z[v][t])
        if z_aux > 5:
            z_aux = 5

        z_level[v][t] = z_aux

        # compare
        assert isinstance(hazard.z[v][t], float)
        assert hazard.z[v][t] == z[v][t]
        assert isinstance(hazard.z_level[v][t], int)
        assert hazard.z_level[v][t] == z_level[v][t]


def test_get():

    g, deadline, d_list = get_param()

    h_list = [1, 2, 3, 4, 5]
    level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    level_color = ['green', 'blue', 'yellow', 'orange', 'red']

    for i in h_list:
        idx = ext.get_python_idx(i)
        hazard = MyHazard(g, deadline, 'fire', i)

        assert hazard.get_level_name(i) == level_label[idx]
        assert hazard.get_level_color(i) == level_color[idx]
        assert hazard.get_level_value(level_label[idx]) == i

    hazard = MyHazard(g, deadline)
    t = 1
    hazard.evolve(t)

    for v in hazard.V:
        assert hazard.get_value_vt(v, t, 0) == hazard.z_level[v][t]
        assert hazard.get_value_vt(v, t, 1) == hazard.z[v][t]
        assert hazard.get_value_vt(v, t, 2) == hazard.z_only[v][t]
        assert hazard.get_value_vt(v, t, 3) == hazard.z_joint[v][t]























