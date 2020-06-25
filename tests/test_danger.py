from core import extract_info as ext
from classes.danger import MyDanger


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
    g, deadline, d_list = get_param()

    danger = MyDanger(g, deadline)

    # test pre defined parameters
    assert danger.levels == [1, 2, 3, 4, 5]
    assert danger.level_label == ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    assert danger.level_color == ['green', 'blue', 'yellow', 'orange', 'red']
    assert danger.n_levels == 5

    assert danger.z.default_z_mu == d_list[0]
    assert danger.z.default_z_sigma == d_list[1]
    assert danger.z.default_f_mu == d_list[2]
    assert danger.z.default_f_sigma == d_list[3]
    assert danger.z.lbda_mu == d_list[4]
    assert danger.z.lbda_sigma == d_list[5]

    # test graph-related parameters
    assert danger.n == 9
    assert danger.V == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert danger.tau == deadline
    assert danger.T == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # coefficients and type
    assert danger.z.type == 'smoke'
    assert danger.z.xi_mu == d_list[0]
    assert danger.z.xi_sigma == d_list[1]

    # z values
    for v in danger.V:
        t = 0
        assert danger.z.z_0[v] == d_list[6]
        assert danger.z.z[v][t] == d_list[6]
        assert danger.z.z_level[v][t] == d_list[6]
        assert danger.z.z_only[v][t] == d_list[6]
        assert danger.z.z_joint[v][t] == 0

    # coefficients and type
    assert danger.f.type == 'fire'
    assert danger.f.xi_mu == d_list[2]
    assert danger.f.xi_sigma == d_list[3]

    # f values
    for v in danger.V:
        t = 0
        assert danger.f.z_0[v] == d_list[6]
        assert danger.f.z[v][t] == d_list[6]
        assert danger.f.z_level[v][t] == d_list[6]
        assert danger.f.z_only[v][t] == d_list[6]
        assert danger.f.z_joint[v][t] == 0


def test_conditional_table():
    g, deadline, d_list = get_param()

    danger = MyDanger(g, deadline)

    value = dict([((1, 1), []), ((3, 3), []), ((5, 3), [])])
    value[(1, 1)] = [0.96, 0.01, 0.01, 0.01, 0.01]
    value[(3, 3)] = [0.01, 0.25, 0.49, 0.24, 0.01]
    value[(5, 3)] = [0.01, 0.01, 0.01, 0.35, 0.62]

    for i in [1, 3, 5]:
        if i == 5:
            j = 3
        else:
            j = i

        my_row = danger.get_row_eta(i, j)

        for k in [1, 2, 3, 4, 5]:
            k_idx = ext.get_python_idx(k)

            my_prob = value[(i, j)][k_idx]

            # check get_eta function
            assert danger.get_eta(i, j, k) == my_prob

            # check get_list_eta function
            assert my_row[k] == my_prob

            # check p_zf
            assert danger.p_zf[(i, j)][k_idx] == my_prob

            # check p_zfd
            assert danger.p_zfd[(i, j, k)] == my_prob


def test_tuple_dicts():

    list1 = [1, 2, 3, 4, 5]
    list2 = [1, 2, 3, 4, 5]
    list3 = [1, 2, 3, 4, 5]

    dict3 = ext.create_3tuple_keys(list1, list2, list3)
    dict2 = ext.create_2tuple_keys(list1, list2)

    assert dict3[(1, 1, 1)] == -1
    assert dict2[(1, 1)] == []
    assert not dict2[(1, 1)]


def test_dependency():
    i = 3
    j = 4

    i1 = MyDanger.check_dependency(i, j)
    i2 = MyDanger.check_dependency(i, j-2)

    assert i1 == j
    assert i2 == i


def test_highest():

    my_list = [1, 2, 3, 4, 5]

    d = ext.create_dict(my_list, 0.5)

    d[2] = 0.2
    d[3] = 0.9
    d[4] = 0.2
    d[5] = 0.1

    p, H = ext.get_highest(d, my_list)

    assert p == 0.9
    assert H == 3


def test_sum():
    g, deadline, d_list = get_param()

    danger = MyDanger(g, deadline)

    var = danger.check_sum_1()

    assert var is True


def test_assign_eta():
    g, deadline, d_list = get_param()
    danger = MyDanger(g, deadline)

    for v in danger.V:
        for t in danger.T:
            f1 = danger.f.z_level[v][t]
            f2 = danger.f.get_value_vt(v, t, 0)

            z1 = danger.z.z_level[v][t]
            z2 = danger.z.get_value_vt(v, t, 0)

            assert f1 == f2
            assert z1 == z2

            z1 = danger.check_dependency(z1, f1)

            eta_row1 = danger.get_row_eta(z1, f1)
            eta_row2 = danger.p_zf[(z1, f1)]

            for k in danger.levels:
                # check assignment
                eta1 = danger.get_eta(z1, f1, k)
                eta2 = danger.p_zfd[(z1, f1, k)]

                assert eta1 == eta2
                assert eta_row1[k] == eta_row2[k-1]

            H1 = danger.H[v][t]
            H2 = ext.get_highest(eta_row1, danger.levels)

            assert H1[0] == H2[0]
            assert H1[1] == H2[1]

            z = z1
            f = f1
            H = H1[1]
            p = H1[0]

            if z == 1:
                if f == 1:
                    assert H == 1
                    assert p == 0.96
            # -----
            if z == 2:
                if f == 1:
                    assert H == 2
                    assert p == 0.96
                if f == 2:
                    assert H == 3
                    assert p == 0.48
            # -----
            if z == 3:
                if f == 1:
                    assert H == 3
                    assert p == 0.48

                if f == 2:
                    assert H == 3
                    assert p == 0.48

                if f == 3:
                    assert H == 3
                    assert p == 0.49
            # -----
            if z == 4:
                if f == 1:
                    assert H == 3
                    assert p == 0.49

                if f == 2:
                    assert H == 4
                    assert p == 0.5

                if f == 3:
                    assert H == 4
                    assert p == 0.55

                if f == 4:
                    assert H == 4
                    assert p == 0.55
            # -----
            if z == 5:
                if f == 1:
                    assert H == 5
                    assert p == 0.5

                if f == 2:
                    assert H == 5
                    assert p == 0.57

                if f == 3:
                    assert H == 5
                    assert p == 0.62

                if f == 4:
                    assert H == 5
                    assert p == 0.72

                if f == 5:
                    assert H == 5
                    assert p == 0.82






