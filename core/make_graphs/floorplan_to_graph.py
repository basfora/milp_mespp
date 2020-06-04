
import matplotlib.pyplot as plt
from core.classes.class_room import MyRoom

from core import extract_info as ext

# CGAL stuff
from CGAL.CGAL_Kernel import Point_2
from CGAL.CGAL_Kernel import Segment_2
from CGAL.CGAL_Kernel import do_intersect
from CGAL.CGAL_Mesh_2 import Mesh_2_Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Mesh_2 import Delaunay_mesh_size_criteria_2
from CGAL.CGAL_Mesh_2 import refine_Delaunay_mesh_2
from CGAL import CGAL_Mesh_2


def ss_1():
    """Parameters of the School Scenario
    MyRoom(label, dim, c0, door)
    label: str
    dim: (x, y)
    c0: (x0, y0)
    door: (x0, y0, 'v/h', size)"""

    n_rooms = 10
    n_doors = 10

    r_list = list(range(1, n_rooms + 1))
    d_list = list(range(1, n_doors + 1))

    school = ext.create_dict(r_list, None)
    doors = ext.create_dict(d_list, None)

    # Gym
    school[1] = MyRoom('Gym', (17, 31), (53, 0))

    doors[1] = {'xy': (67.4, 31.), 'o': 0, 'size': 0.9}
    # school[1].place_door(doors[1])

    # Hall
    H1 = MyRoom('H', (3.5, 12.4), (66.5, 31))
    H1.merge((60, 4), (10, 43.4))
    H1.merge((10, 4.3), (10, 47.4))
    school[2] = H1

    # small rooms
    school[3] = MyRoom('A', (5.8, 3), (60.7, 32.2))
    doors[3] = {'xy': (66.5, 33), 'o': 1, 'size': 0.9}

    school[4] = MyRoom('B', (5.8, 3), (60.7, 35.8))
    doors[4] = {'xy': (66.5, 36.6), 'o': 1, 'size': 0.9}

    school[5] = MyRoom('C', (5.8, 3), (60.7, 39.4))
    doors[5] = {'xy': (66.5, 40.2), 'o': 1, 'size': 0.9}

    school[6] = MyRoom('D', (8, 5.5), (60.5, 47.4))
    doors[6] = {'xy': (66.8, 47.4), 'o': 0, 'size': 0.9}

    school[7] = MyRoom('E', (8, 5.5), (51.1, 47.4))
    doors[7] = {'xy': (57.4, 47.4), 'o': 0, 'size': 0.9}

    school[8] = MyRoom('F', (8, 5.5), (41.7, 47.4))
    doors[8] = {'xy': (48.0, 47.4), 'o': 0, 'size': 0.9}

    school[9] = MyRoom('G', (8, 5.5), (32.3, 47.4))
    doors[9] = {'xy': (38.6, 47.4), 'o': 0, 'size': 0.9}

    school[10] = MyRoom('Cafe', (10, 17), (0, 33.5))
    doors[10] = {'xy': (10, 46.7), 'o': 1, 'size': 0.9}

    return school


def house_hri():
    """label: str
    dim: (x, y)  c0: (x0, y0)"""
    n_rooms = 24
    r_list = list(range(1, n_rooms + 1))
    house = ext.create_dict(r_list, None)

    house[1] = MyRoom('Garage', (5.1, 5.8), (1.7, -10.4))
    house[2] = MyRoom('Lounge', (3.8, 3.7), (1.9, -4.4))
    house[3] = MyRoom('Living', (3.8, 5.2), (2.0, -0.5))
    house[4] = MyRoom('Dining', (2.7, 6.6), (3.1, 4.9))
    house[5] = MyRoom('Porch', (1.5, 1.7), (-0.1, -11.7))
    house[6] = MyRoom('Coat', (1.7, 2.9), (-2.1, -10.9))
    house[7] = MyRoom('WC1', (1.1, 0.8), (-1.1, -6))
    house[8] = MyRoom('PDR', (1.9, 1.2), (-1.8, -5))
    house[9] = MyRoom('Pantery', (1.6, 2.0), (-2.2, -2.3))
    house[10] = MyRoom('Rumpus', (4, 3.9), (-1.1, 7.7))
    house[11] = MyRoom('Master BD', (4.1, 4), (-6.5, -10.2))
    house[12] = MyRoom('Study', (4.1, 2.8), (-6.5, -3.7))
    house[13] = MyRoom('BD2', (4, 2.9), (-6.5, -0.7))
    house[14] = MyRoom('BD3', (3.2, 3.4), (-6.5, 2.4))
    house[15] = MyRoom('Bath 1', (2.9, 1.9), (-6.5, 6.0))
    house[16] = MyRoom('BD4', (3.1, 3.4), (-6.5, 8.1))
    house[17] = MyRoom('Laundry', (1.9, 2.2) , (-3.2, 9.3))
    house[18] = MyRoom('Bath2', (1.4, 0.9), (-2.7, -6))
    house[19] = MyRoom('Kitchen', (3.9, 7.6), (-2.2, -0.1))
    house[20] = MyRoom('Master Bath', (3.6, 2.1), (-6.5, -6.0))
    # halls
    house[21] = MyRoom('Hall 1', (1.5, 3.5), (-0.1, -9.8))
    house[22] = MyRoom('Hall 2', (1.3, 5.5), (0.3, -5.8))
    house[23] = MyRoom('Hall 3', (1, 2.3), (1.9, 5.1))
    house[24] = MyRoom('Hall 4', (0.8, 6.0), (-3.2, 2.5))

    return house



def school_not_elegant():
    # points
    c = dict()
    # gym, H1, H2 (right)
    c[1], c[2], c[3] = (53., 0.), (70., 0.), (70.0, 47.4)
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
    c[28], c[29], c[30], c[31], c[32] = (60.7,	43.4), (66.5,	42.4), (60.7,	42.4), (66.5,	39.4), (56.6,	39.4)
    # B
    c[33], c[34], c[35], c[36] = (60.7,	38.8), (66.5,	38.8), (66.5,	35.8), (60.7,	35.8)
    # A
    c[37], c[38], c[39], c[40] = (60.7,	35.2), (66.5,	35.2), (66.5,	32.2), (60.7,	32.2)
    #
    c[41], c[42] = (56.6,	31.0), (53.0,	31.0)
    # DOOR gym
    # c[43], c[44] = (43.0,	3.8), (43.0, 2.0)
    # # inside doors H1
    # c[45], c[46] = (57.4,	31.0),	(58.3,	31.0)
    # # A
    # c[47], c[48] = (56.6,	33.0), (56.6,	33.9)
    # # B
    # c[49], c[50] = (56.6,	36.6), (56.6,	37.5)
    # # C
    # c[51], c[52] = (56.6,	40.2), (56.6,	41.1)
    # # D
    # c[53], c[54] = (66.8,	47.4), (67.7,	47.4)
    # # E
    # c[55], c[56] = (57.4,	47.4), (58.3,	47.4)
    # # F
    # c[57], c[58] = (48.0,	47.4), (48.9,	47.4)
    # # G
    # c[59], c[60] = (38.6,	47.4), (39.5,	47.4)
    #
    # c[61] = (70.0,	31.0)
    # # cafe
    # c[62], c[63] = (10.0,	46.7), (10.0,	47.6)

    wall_nodes = []
    wall_conn = []
    for i in c.keys():
        wall_nodes.append(c[i])
        idx = i-1
        if i < 42:
            wall_conn.append((idx, idx+1))
        if i == 42:
            wall_conn.append((idx, 0))


    # doors
    # gym
    # wall_conn.append((42-1, 43-1))
    # wall_conn.append((44-1, 1-1))
    # wall_conn.append((44-1, 43-1))
    # # H1
    # wall_conn.append((41-1, 45-1))
    # wall_conn.append((46-1, 61-1))
    # # A
    # wall_conn.append((40-1, 47-1))
    # wall_conn.append((48-1, 37-1))
    # # B
    # wall_conn.append((36-1, 49-1))
    # wall_conn.append((50-1, 33-1))
    # # C
    # wall_conn.append((32-1, 51-1))
    # wall_conn.append((52-1, 29-1))
    # # D
    # wall_conn.append((4-1, 54-1))
    # wall_conn.append((7-1, 53-1))
    # # E
    # wall_conn.append((8-1, 56-1))
    # wall_conn.append((11-1, 55-1))
    # # F
    # wall_conn.append((15-1, 57-1))
    # wall_conn.append((12-1, 58-1))
    # # G
    # wall_conn.append((16-1, 60-1))
    # wall_conn.append((19-1, 59-1))
    # # cafe
    # wall_conn.append((27 - 1, 62 - 1))
    # wall_conn.append((22 - 1, 63 - 1))

    print(max(c.keys()))
    print(len(wall_nodes))
    print(len(wall_conn))

    return wall_nodes, wall_conn



def ss_1_merged():

    n_rooms = 10
    r_list = list(range(1, n_rooms + 1))
    school = ext.create_dict(r_list, None)

    school[1] = MyRoom('Gym', (17, 31), (53, 0))


# ---------------------------------------------------------------------------------
# organizing data
# ----------------------------------------------------------------------------------

def nodes_and_conections(scenario: dict):
    """Loop through scenario and add nodes and connections to lists
    wall_nodes = [(x1, y1), (x2, y2)...]
    wall_conn = [(p1, p2)..]"""

    wall_nodes = []
    wall_conn = []

    for k in scenario.keys():

        r = scenario[k]
        # keep track of existent nodes
        i = len(wall_nodes)

        # get corner points from each room
        p = r.c

        # append to list
        wall_nodes = wall_nodes + p

        n_c = len(p) - 1
        # walls: 0-1, 1-2, 2-3, 3-1
        for j in range(n_c):
            wall_conn.append((i+j, i+j+1))
        wall_conn.append((i, i+n_c))

    return wall_nodes, wall_conn


# ---------------------------------------------------------------------------------
# simple CGAL functions that I can never remember
# ----------------------------------------------------------------------------------

def add_point(cdt, p):
    v = cdt.insert(Point_2(p[0], p[1]))
    return v


def add_wall(cdt, va, vb):
    cdt.insert_constraint(va, vb)


def create_cdt():
    cdt = Mesh_2_Constrained_Delaunay_triangulation_2()

    return cdt


def get_vertex(p):
    v = (round(p.x(), 2), round(p.y(), 2))
    print(v)
    return v


def return_as_list(points):
    V = []
    for p in points:
        v = get_vertex(p)
        V.append(v)

    return V

# ---------------------------------------------------------------------------------
def plot_floorplan(scenario: dict, V=None, door=False, middle=False):

    for k in scenario.keys():

        r = scenario[k]

        x = [r.c1[0], r.c2[0], r.c3[0], r.c4[0], r.c1[0]]
        y = [r.c1[1], r.c2[1], r.c3[1], r.c4[1], r.c1[1]]

        plt.plot(x, y, 'k-')

        if middle is True:
            x_m = (x[1] + x[3]) / 2
            y_m = (y[1] + y[3]) / 2

            plt.plot(x_m, y_m, 'green', linewidth=3)

        if door is True and r.has_door is True:
            door_color = '#f1f1f1'
            if r.label == 'H3':
                door_color = 'w'
            # plot door as light gray line
            x_door = [r.door[0], r.door[0] + r.door_delta[0]]
            y_door = [r.door[1], r.door[1] + r.door_delta[1]]
            plt.plot(x_door, y_door, door_color, linewidth=3)

    if V is not None:
        for i in range(len(V)):
            xv = V[i][0]
            yv = V[i][1]
            plt.plot(xv, yv, 'ro', linewidth=1)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('School Scenario (SS-1)')

    folder_name = "/home/basfora/Insync/beatriz.asfora@gmail.com/Google Drive/Campbell Lab/08. Undegrad Project/"
    fig_name = "School_delauney"
    my_format = ".png"

    fname = folder_name + fig_name + my_format

    plt.savefig(fname, facecolor=None, edgecolor=None,
                orientation='landscape', papertype=None,
                transparent=True)
    plt.show()

    return plt


def mesh_area(scenario: dict):

    # area = Constrained_Delaunay_triangulation_2()

    cdt = Mesh_2_Constrained_Delaunay_triangulation_2()

    for k in scenario.keys():
        r = scenario[k]

        # 2D points =  corners
        va = cdt.insert(Point_2(r.c1[0], r.c1[1]))
        vb = cdt.insert(Point_2(r.c2[0], r.c2[1]))
        vc = cdt.insert(Point_2(r.c3[0], r.c3[1]))
        vd = cdt.insert(Point_2(r.c4[0], r.c4[1]))

        # constraints = walls
        cdt.insert_constraint(va, vb)
        cdt.insert_constraint(vb, vc)
        cdt.insert_constraint(vc, vd)
        cdt.insert_constraint(vd, va)

    print("Number of vertices before: ", cdt.number_of_vertices())

    # make first mesh
    # CGAL_Mesh_2.make_conforming_Delaunay_2(area)

    # angle, side
    CGAL_Mesh_2.refine_Delaunay_mesh_2(cdt, Delaunay_mesh_size_criteria_2(0.125, 4.))

    print("Number of vertices after make_conforming_Delaunay_2: ", cdt.number_of_vertices())

    # then make it conforming Gabriel
    # CGAL_Mesh_2.make_conforming_Gabriel_2(area)

    # print("Number of vertices after make_conforming_Gabriel_2: ", area.number_of_vertices())

    V = return_as_list(cdt.points())

    return V


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


def is_not_intersect_boundary(wall_nodes, wall_conn, c0, c1):
    seg_1 = Segment_2(Point_2(c0[0], c0[1]), Point_2(c1[0], c1[1]))

    for wall_k in wall_conn:
        p0, p1 = get_wall_seg_ends(wall_nodes, wall_k)
        seg_2 = Segment_2(p0, p1)
        if do_intersect(seg_1, seg_2):
            return False

    return True


def get_wall_seg_ends(wall_nodes, wall_k):
    i0 = wall_k[0]
    i1 = wall_k[1]
    n0 = wall_nodes[i0]
    n1 = wall_nodes[i1]
    p0 = Point_2(n0[0], n0[1])
    p1 = Point_2(n1[0], n1[1])
    return p0, p1


def get_tri_vertices(cdt):
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

    return DT_points, DT_conn


def create_mesh_refine(wall_nodes, wall_conn):
    cdt = create_cdt()

    for wall_k in wall_conn:
        p0, p1 = get_wall_seg_ends(wall_nodes, wall_k)
        cdt.insert_constraint(p0, p1)

    refine_Delaunay_mesh_2(cdt, Delaunay_mesh_size_criteria_2(0.125, 10.0))

    return cdt


def create_voronoi(cdt, wall_nodes, wall_conn):
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
                    if is_not_intersect_boundary(wall_nodes, wall_conn, c0, cj):
                        i_k = all_points.index(cj)
                        if i < i_k:
                            all_connect.append((i, i_k))

    return all_points, all_connect


def generate_file(all_points):

    V = all_points

    f = open('school_2.txt', 'w+')

    line1 = '{0:4s} {1:8s} \n'.format('SS-1', '0-0 BL')
    line2 = '{0:4s} {1:8s} {2:6s} \n'.format('ID', 'X', 'Y')

    f.write(line1)
    f.write(line2)

    for v in V:
        v_idx = V.index(v) + 1
        x = v[0]
        y = v[1]

        my_text = '{0:3d} {1:8.4f} {2:8.4f} \n'.format(v_idx, x, y)
        f.write(my_text)

    # for line in f:
    #     my_data = line.split()  # Splits on whitespace
    #     print(repr(my_data[0]).rjust(2), repr(my_data[1]).rjust(3), end=' ')
    #     print(repr(my_data[3].rjust(4)))

        # print('{0:2s} {1:3s} {2:4s}'.format(my_data[0], my_data[1], my_data[2]))

    f.close()


def plot_hri(scenario: dict, V=None, middle=False):

    my_color = 'k'
    for k in scenario.keys():

        r = scenario[k]

        x, y = [], []

        for i in range(len(r.c)):
            x.append(r.c[i][0])
            y.append(r.c[i][1])

        x.append(r.c[0][0])
        y.append(r.c[0][1])

        if k > 20:
            my_color = 'b'

        plt.plot(x, y, my_color)

        plt.text(x[0] + 0.1, y[2]-0.6, str(k))

        if middle is True:
            x_m = (x[1] + x[3]) / 2
            y_m = (y[1] + y[3]) / 2

            plt.plot(x_m, y_m, 'go', linewidth=2)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Home Rooms')

    plt.axis('equal')

    folder_name = "/home/basfora/Insync/beatriz.asfora@gmail.com/Google Drive/2020.1/HRI/01. Final Project/floorplan/"
    fig_name = "hri"
    my_format = ".pdf"

    fname = folder_name + fig_name + my_format

    plt.savefig(fname, facecolor=None, edgecolor=None,
                orientation='landscape', papertype=None,
                transparent=True)
    plt.show()

    return plt


def main_function():

    # set up scenario
    school = ss_1()
    wall_nodes, wall_conn = nodes_and_conections(school)

    # wall_nodes, wall_conn = school_not_elegant()

    cdt = create_mesh_refine(wall_nodes, wall_conn)

    # plot dt
    # DT_points, DT_conn = get_tri_vertices(cdt)
    plot_points_between_list(wall_nodes, wall_conn, 'k')
    # plt.show()

    # Construct Voronoi
    all_points, all_connect = create_voronoi(cdt, wall_nodes, wall_conn)

    print(len(all_connect))
    plot_points_between_list(all_points, all_connect, 'g')
    print(len(all_points))
    # plt.show()

    generate_file(all_points)
    plt.show()


if __name__ == '__main__':
    main_function()






