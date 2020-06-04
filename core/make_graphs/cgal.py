

from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Kernel import Segment_2
from CGAL.CGAL_Kernel import do_intersect
from CGAL.CGAL_Mesh_2 import Mesh_2_Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Mesh_2 import Delaunay_mesh_size_criteria_2
from CGAL.CGAL_Mesh_2 import refine_Delaunay_mesh_2

import matplotlib.pyplot as plt


def centroid(t):
    pv0 = t.vertex(0)
    pv1 = t.vertex(1)
    pv2 = t.vertex(2)
    px = (pv0.x() + pv1.x() + pv2.x()) / 3
    py = (pv0.y() + pv1.y() + pv2.y()) / 3
    pc = (px, py)
    return pc


def plot_points_between_list(v_points, v_conn, color):
    for wall_k in v_conn:
        i0 = wall_k[0]
        i1 = wall_k[1]
        n0 = v_points[i0]
        n1 = v_points[i1]
        px = [n0[0], n1[0]]
        py = [n0[1], n1[1]]
        plt.plot(px, py, color + '.-')
    return None


def is_not_intersect_boundary(c0, c1):
    seg_1 = Segment_2(Point_2(c0[0], c0[1]), Point_2(c1[0], c1[1]))

    for wall_k in wall_conn:
        p0, p1 = get_wall_seg_ends(wall_k)
        seg_2 = Segment_2(p0, p1)
        if do_intersect(seg_1, seg_2):
            return False

    return True


def get_wall_seg_ends(wall_k):
    i0 = wall_k[0]
    i1 = wall_k[1]
    n0 = wall_nodes[i0]
    n1 = wall_nodes[i1]
    p0 = Point_2(n0[0], n0[1])
    p1 = Point_2(n1[0], n1[1])
    return p0, p1


def school_not_elegant():
    # points
    c = dict()
    # gym, H1, H2 (right)
    c[1], c[2], c[3] = (43., 0.), (70., 0.), (70.0, 47.4)
    # D, E, F, G
    c[4], c[5], c[6], c[7] = (68.5, 47.4), (68.5, 52.9), (60.5, 52.9), (60.5, 47.4)
    c[8], c[9], c[10], c[11] = (59.1, 47.4), (59.1, 52.9), (51.1, 52.9), (51.1, 47.4)
    c[12], c[13], c[14], c[15] = (49.7, 47.4), (49.7, 52.9), (41.7, 52.9), (41.7, 47.4)
    c[16], c[17], c[18], c[19] = (40.3, 47.4), (40.3, 52.9), (32.3, 52.9), (32.3, 47.4)
    # H3
    c[20], c[21], c[22] = (20.0, 47.4), (20.0, 51.7), (10.0, 51.7)
    # cafe
    c[23], c[24], c[25], c[26], c[27] = (10.0, 50.5), (0.0, 50.5), (0.0, 33.5), (10.0, 33.5), (10.0, 43.4)
    # C
    c[28], c[29], c[30], c[31], c[32] = (56.6,	43.4), (56.6,	42.4), (50.8,	42.4), (50.8,	39.4), (56.6,	39.4)
    # B
    c[33], c[34], c[35], c[36] = (56.6,	38.8), (50.8,	38.8), (50.8,	35.8), (56.6,	35.8)
    # A
    c[37], c[38], c[39], c[40] = (56.6,	35.2), (50.8,	35.2), (50.8,	32.2), (56.6,	32.2)
    #
    c[41], c[42] = (56.6,	31.0), (43.0,	31.0)
    # DOOR gym
    c[43], c[44] = (43.0,	3.8), (43.0, 2.0)
    # inside doors H1
    c[45], c[46] = (57.4,	31.0),	(58.3,	31.0)
    # A
    c[47], c[48] = (56.6,	33.0), (56.6,	33.9)
    # B
    c[49], c[50] = (56.6,	36.6), (56.6,	37.5)
    # C
    c[51], c[52] = (56.6,	40.2), (56.6,	41.1)
    # D
    c[53], c[54] = (66.8,	47.4), (67.7,	47.4)
    # E
    c[55], c[56] = (57.4,	47.4), (58.3,	47.4)
    # F
    c[57], c[58] = (48.0,	47.4), (48.9,	47.4)
    # G
    c[59], c[60] = (38.6,	47.4), (39.5,	47.4)

    c[61] = (70.0,	31.0)
    # cafe
    c[62], c[63] = (10.0,	46.7), (10.0,	47.6)

    wall_nodes = []
    wall_conn = []
    for i in c.keys():
        wall_nodes.append(c[i])
        idx = i-1
        if i < 43:
            wall_conn.append((idx, idx+1))

    # doors
    # gym
    wall_conn.append((42-1, 43-1))
    wall_conn.append((44-1, 1-1))
    wall_conn.append((44-1, 43-1))
    # H1
    wall_conn.append((41-1, 45-1))
    wall_conn.append((46-1, 61-1))
    # A
    wall_conn.append((40-1, 47-1))
    wall_conn.append((48-1, 37-1))
    # B
    wall_conn.append((36-1, 49-1))
    wall_conn.append((50-1, 33-1))
    # C
    wall_conn.append((32-1, 51-1))
    wall_conn.append((52-1, 29-1))
    # D
    wall_conn.append((4-1, 54-1))
    wall_conn.append((7-1, 53-1))
    # E
    wall_conn.append((8-1, 56-1))
    wall_conn.append((11-1, 55-1))
    # F
    wall_conn.append((15-1, 57-1))
    wall_conn.append((12-1, 58-1))
    # G
    wall_conn.append((16-1, 60-1))
    wall_conn.append((19-1, 59-1))
    # cafe
    wall_conn.append((27 - 1, 62 - 1))
    wall_conn.append((22 - 1, 63 - 1))

    print(max(c.keys()))
    print(len(wall_nodes))
    print(len(wall_conn))

    return wall_nodes, wall_conn


# wall_nodes = [(0.1, 0.1), (5.1, 0.1), (10.0, 0.1), (10.0, 5.1), (5.1, 5.1), (5.1, 10.0), (0.1, 10.0), (0.1, 5.1), (5.1, 2.1), (5.1, 0.1)]
# wall_conn = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (9, 4), (4, 9)]

wall_nodes, wall_conn = school_not_elegant()


# CGAL stuffs
cdt = Mesh_2_Constrained_Delaunay_triangulation_2()
for wall_k in wall_conn:
    p0, p1 = get_wall_seg_ends(wall_k)
    cdt.insert_constraint(p0, p1)

refine_Delaunay_mesh_2(cdt, Delaunay_mesh_size_criteria_2(0.02, 50.0))


## PLOT DT
DT_points = []
DT_conn = []
for face_k in cdt.finite_faces():
    t = cdt.triangle(face_k)
    pv0 = t.vertex(0)
    pv1 = t.vertex(1)
    pv2 = t.vertex(2)

    i = DT_points.__len__()

    DT_points.append((pv0.x(), pv0.y()))
    DT_points.append((pv1.x(), pv1.y()))
    DT_points.append((pv2.x(), pv2.y()))

    DT_conn.append((i + 0, i + 1))
    DT_conn.append((i + 1, i + 2))
    DT_conn.append((i + 2, i + 0))


# plot_points_between_list(DT_points, DT_conn, 'b')

plot_points_between_list(wall_nodes, wall_conn, 'r')


### Construct Voronoi
all_points = []
all_connect = []
for face_k in cdt.finite_faces():
    tri_k = cdt.triangle(face_k)
    c0 = centroid(tri_k)
    if face_k.is_in_domain():
        all_points.append(c0)

for face_k in cdt.finite_faces():
    if face_k.is_in_domain():

        tri_k = cdt.triangle(face_k)
        c0 = centroid(tri_k)

        i = all_points.index(c0)
        for j in range(3):
            n0 = face_k.neighbor(j)
            cj = centroid(cdt.triangle(n0))
            if n0.is_in_domain():
                if is_not_intersect_boundary(c0, cj):
                    i_k = all_points.index(cj)
                    if i < i_k:
                        all_connect.append((i, i_k))

print(len(all_connect))

plot_points_between_list(all_points, all_connect, 'g')

print(len(all_points))

plt.show()













