# -----------------------------
# make graph from floorplan
# input:
# output: txt file
# TODO plot vertices
# ----------------------------
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

# compute Delaunay
points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
# takes array of points
tri = Delaunay(points)

# visualize
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')


# pretty plot
for j, p in enumerate(points):
    plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points
for j, s in enumerate(tri.simplices):
    p = points[s].mean(axis=0)
    plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.show()



# -----------------------------------------------------------------------------------------------------------
# from CGAL.CGAL_Kernel import Point_2
# from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
# from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2_Vertex_handle
# from CGAL import CGAL_Mesh_2
# from CGAL.CGAL_Mesh_2 import Delaunay_mesh_size_criteria_2
#
#
#
# def mesh_area():
#     area1 = Constrained_Delaunay_triangulation_2()
#
#     x = [53., 70.]
#     y = [0., 31.]
#
#     # coordinates gym
#     # bottom
#     x_bl, y_bl = x[0], y[0]
#     x_br, y_br = x[1], y[0]
#
#     # top
#     x_tl, y_tl = x[0], y[1]
#     x_tr, y_tr = x[1], y[1]
#
#     # constrained triangulation
#     # 2D points =  corners
#     va = area1.insert(Point_2(x_bl, y_bl))
#     vb = area1.insert(Point_2(x_br, y_br))
#     vc = area1.insert(Point_2(x_tl, y_tl))
#     vd = area1.insert(Point_2(x_tr, y_br))
#
#     # constraints = walls
#     area1.insert_constraint(va, vb)
#     area1.insert_constraint(vb, vc)
#     area1.insert_constraint(vc, vd)
#     area1.insert_constraint(vd, va)
#
#     my_x = [x_bl, x_br, x_tr, x_tl, x_bl]
#     my_y = [y_bl, y_br, y_tr, y_tl, y_bl]
#
#     for i in range(len(my_x)-1):
#
#         x1, y1 = my_x[i], my_y[i]
#         x2, y2 = my_x[i + 1], my_y[i + 1]
#
#         plt.plot([x1, x2], [y1, y2], 'bo-')
#         # plt.show()
#
#         print("(x1, y1): %d, %d (x2, y2): %d, %d" % (x1, y1, x2, y2))
#
#     plt.plot(my_x, my_y, 'ro-')
#     plt.show()
#
#
#
#     # make first mesh
#     CGAL_Mesh_2.make_conforming_Delaunay_2(area1)
#
#     print("Number of vertices before: ", area1.number_of_vertices())
#     for p in area1.points():
#         print(p)
#         pp = dir(p)
#
#         # r1 = p.this()
#         r2 = p.x()
#
#         print(pp)
#
#         True
#
#     plt.show()
#     print('-----')
#
#     # make it conforming Delaunay
#     CGAL_Mesh_2.make_conforming_Delaunay_2(area1)
#
#     # area criteria ?
#     # CGAL_Mesh_2.refine_Delaunay_mesh_2(area1, Delaunay_mesh_size_criteria_2(0.125, 0.5))
#
#     print("Number of vertices after make_conforming_Delaunay_2: ", area1.number_of_vertices())
#
#     # then make it conforming Gabriel
#     CGAL_Mesh_2.make_conforming_Gabriel_2(area1)
#
#     print("Number of vertices after make_conforming_Gabriel_2: ", area1.number_of_vertices())
#     for p in area1.points():
#         print(p)
#
#
#
#
# # --------
# mesh_area()
#
#
#
#
#
#
