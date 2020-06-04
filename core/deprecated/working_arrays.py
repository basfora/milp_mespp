import core.extract_info as ext
from core import create_parameters as cp


def parameters_7v_random_motion():
    """Parameters pre-defined for unit tests"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = core.extract_info.get_graph(graph_file)
    # input for target initial vertices (belief)
    v_target = [7]
    # initial searcher vertices
    v_searchers = [1, 2]
    deadline = 3
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    # initialize parameters
    b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution)
    n = 7
    return n, b_0, M, searchers_info

# # input parameters for graph
# n_vertex = 7
#
# graph_file = 'G_7V7E.p'
# g = cm.get_graph(graph_file)
#
# # input for target initial vertices (belief)
# v_target = [7]
#
# # type of motion
# target_motion = 'random'
# belief_distribution = 'uniform'
# b_0, M = cp.my_target_motion(g, v_target, belief_distribution, target_motion)
#
# big_M = ac.assemble_big_matrix(n_vertex, M)
# print("done")
# #
# my_vertices = [1, 2, 3, 4, 5, 6, 7]
# prob_move = [0.5, 0.5, 0, 0, 0, 0, 0]
# count1 = 0
# count2 = 0
# count3 = 0
# N = 10000
# i = 0
# while i < N:
#     my_vertex = ac.sample_vertex(my_vertices, prob_move)
#     if my_vertex == 1:
#         count1 = count1 + 1
#     elif my_vertex == 2:
#         count2 = count2 + 1
#     else:
#         count3 += count3
#     i = i + 1
#
# print(count1)
# print(count2)

# # load graph
# graph_file = 'G_7V7E.p'
# g = cm.get_graph(graph_file)
# # input for target initial vertices (belief)
# v_target = [7]
# # initial searcher vertices
# v_searchers = [1, 2]
# deadline = 3
# # type of motion
# target_motion = 'random'
# belief_distribution = 'uniform'
# b_0, M = cp.my_target_motion(g, v_target, belief_distribution, target_motion)
# searchers_info = cp.my_searchers_info(g, v_searchers)
#
# searchers_pos = {}
# searchers_pos[(1, 0)] = 1
# searchers_pos[(2, 0)] = 2
#
#
# prod_C = ac.product_capture_matrix(searchers_info, searchers_pos, 7, 0)

# n, b_0, M, searchers_info = parameters_7v_random_motion()
#
# b = MyBelief(b_0)
#
# print(b.stored_belief)
# print(b.init_belief)
#
# print(b.stored_belief.get(0))
#
# m = len(b.init_belief)
#
# print(list(range(1, m)))

#
# v_target = [6, 7]
#
# n, b_0, M, searchers_info = parameters_7v_random_motion()
#
# target = MyTarget(v_target, M)


def parameters_7v_random_motion2():
    """Parameters pre-defined for unit tests"""
    # load graph
    graph_file = 'G_7V7E.p'
    g = core.extract_info.get_graph(graph_file)
    # input for target initial vertices (belief)
    v_target = [7]
    # initial searcher vertices
    v_searchers = [1, 2]
    deadline = 3
    # type of motion
    target_motion = 'random'
    belief_distribution = 'uniform'
    # initialize parameters
    b_0, M, searchers_info = cp.init_parameters(g, v_target, v_searchers, target_motion, belief_distribution)
    n = 7
    return n, b_0, M, searchers_info, g

#
# horizon = 3
# n, b_0, M, searchers_info, g = parameters_7v_random_motion2()
# # solve
# # create model
# md = Model("my_model")
#
# my_vars = mf.add_variables(md, g, searchers_info, horizon)
#
# mf.add_constraints(md, g, my_vars, searchers_info, horizon, b_0, M)
#
# mf.set_obj_fcn(md, 1.5, horizon, my_vars)
#
# md.update()
# # Optimize model
# md.optimize()
#
# s_pos, b_target = ar.query_variables(md, g, searchers_info, horizon)
#
# obj_fun = md.objVal
# gap = md.MIPGap
# time_sol = round(md.Runtime, 4)
# my_data = MySolverData(obj_fun, time_sol, gap, s_pos, b_target, horizon)


#
#
# t, theta = 1, 2
# print(t % theta)

# filename = "/home/basfora/PycharmProjects/multiRobotTargetSearch/data/CH10S1G2TNMV_0930_001" + "/" + 'global_save.txt'
# exp_data = sf.load_pickle_file(filename)
# #
# target = exp_data['target']
# searchers = exp_data['searchers']
# s = searchers[1]
# print(target.seed)
# print(s.seed)


#for i in range(20):
#    my_r = np.random.random()
#    if my_r <= 1-0.3:
#        print(my_r)

#print(datetime.datetime.today().day + 1)
# print(exp_data['solver_data'].gap)
#
#
# g = pmi.create_papers_graph()
# g.vs["color"] = "white"
# plot(g, 'G60V_2.pdf')
# print(sys.path)
my_seed = 2000
my_list = list(range(1, 60))
for i in range(20):
    my_seed = ext.get_random_seed()
    my_random_list = cp.pick_pseudo_random(my_list, my_seed, 5)
    print(my_random_list)

# import random
#
# print(round(datetime.datetime.now().timestamp() * 100))
#
# for i in range(20):
#     ext.get_random_seed()
#     print(random.random())