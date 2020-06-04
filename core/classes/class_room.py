"""Class for each room os school scenario
Rooms are assumed to be rectangular"""


class MyRoom:

    def __init__(self, label: str, dim: tuple, c0: tuple, door=None):

        # name of the room
        self.label = label
        # dimensions (x, y)
        self.dim = dim
        # bottom left coordinates (x0, y0)
        self.c1 = c0
        self.type = None
        # other coordinates (assuming rectangular shape)
        self.c2 = (0., 0.)
        self.c3 = (0., 0.)
        self.c4 = (0., 0.)

        self.c = []
        self.c_all = []

        self.conn = []

        self.has_door = False
        self.door = dict()
        self.d_size = 0.9
        self.d_c = [(0., 0), (0., 0.)]
        self.d_delta = (0., 0.)

        # if door is not None:
        #     self.place_door(door)

        # room area
        self.area = 0.0

        self.set_coordinates()
        self.set_area()

    def set_coordinates(self):
        c = self.get_coordinates(self.dim, self.c1)
        self.c = c

        self.conn = self.get_connections(c)


    @staticmethod
    def get_connections(sorted_c: list):

        wall_conn = []
        for i in range(len(sorted_c)-1):
            wall_conn.append((i, i+1))
        wall_conn.append((0, i+1))

        return wall_conn

    @staticmethod
    def get_coordinates(dim: tuple, c1: tuple):
        """c: bottom left coordinates (x, y)
          dim: dimensions of room (x,y)"""

        # unpack
        x0, y0 = c1[0], c1[1]
        dx, dy = dim[0], dim[1]

        # bottom left (given) / right
        c1 = (x0, y0)
        c2 = (x0 + dx, y0)

        # top left/right
        c4 = (x0, y0 + dy)
        c3 = (x0 + dx, y0 + dy)

        c = list()
        c.append(c1)
        c.append(c2)
        c.append(c3)
        c.append(c4)

        # for i in range(len(c)):
        #     c[i] = (round(c[i][0], 4), round(c[i][0], 4))

        return c

    @staticmethod
    def get_area(dim: tuple):

        x = dim[0]
        y = dim[1]

        area = x*y

        return area

    def set_area(self):
        self.area = self.get_area(self.dim)

    # def place_door(self, door: dict):
    #
    #     self.has_door = True
    #
    #     self.d_size = door['size']
    #
    #     if door['o'] is 1:
    #         self.d_delta = (0., self.d_size)
    #     else:
    #         self.d_delta = (self.d_size, 0)
    #
    #     c0 = door['xy']
    #     c1 = (c0[0] + self.d_delta[0], c0[1] + self.d_delta[1])
    #
    #     self.d_c = [c0, c1]
    #     self.door = door
    #
    #     self.break_for_door(door)

    # def break_for_door(self, door=None):
    #
    #     i = 0
    #     p1 = self.d_c[0]
    #     while True:
    #         p2 = self.c[i]
    #         if self.connected(p1, p2) is True:
    #             break
    #         i += 1
    #
    #     # insert door nodes (sorted)
    #     aux_c = self.c
    #     i1 = i + 1
    #     i2 = i + 2
    #
    #     aux_c.insert(i1, self.d_c[0])
    #     aux_c.insert(i2, self.d_c[1])
    #
    #     # fix connections
    #     conn_list = self.conn
    #
    #     aux_list = conn_list[:i1]
    #
    #     i3 = conn_list[i2]
    #
    #     aux_list.append((i2, i3))
    #
    #     k = 0
    #     for con in conn_list[i1:]:
    #         if con[0] > i:
    #             conn_list[k][0] = con[0] + 2
    #         if con[1] > i:
    #             conn_list[k][1] = con[1] + 2
    #         k += 1
    #
    #     print(conn_list)
    #
    #
    #
    #
    #     i = aux_c.index(self.d_c[0])
    #     j = aux_c.index(self.d_c[1])
    #
    #     conn_list.insert(i, (i - 1, i))
    #     conn_list.insert(j, (j, j + 1))
    #
    #     #self.add_connection_door(conn_list, i, j)
    #
    #     self.c_all = aux_c

    def merge(self, dim: tuple, c0: tuple):

        c_r1 = self.c
        area_r1 = self.area

        c_r2 = self.get_coordinates(dim, c0)
        area_r2 = self.get_area(dim)

        # update room area
        new_area = area_r1 + area_r2
        self.area = new_area

        # take out repeated vertices
        new_c = self.new_points(c_r1, c_r2)
        # find extreme x and y
        pp = self.extreme_points(new_c)

        # put in order (counterclockwise)
        sorted_c = [new_c[0]]
        aux_c = new_c[1:]

        i, ll = 0,  len(aux_c)
        last_point = sorted_c[0]

        while True:

            if ll == 0:
                break

            # find the next connected point on the list
            point = aux_c[i]
            # print('last: %s, point: %s' % (last_point, point))

            if self.connected(last_point, point) is False:
                i += 1
                continue

            if point not in sorted_c:
                sorted_c.append(point)
                aux_c.remove(point)

                # iterate
                last_point = point
                ll = len(aux_c)

            if ll <= i:
                i = 0

        # print(sorted_c)

        self.c = sorted_c

    @staticmethod
    def iterate(ll, i):

        if i == ll - 1:
            i = 0
        else:
            i += 1

        return i

    @staticmethod
    def connected(point1, point2):
        if point1[0] == point2[0] or point1[1] == point2[1]:
            return True
        else:
            return False

    @staticmethod
    def update_xy(points_list):
        y, x = [], []

        for point in points_list:
            x.append(point[0])
            y.append(point[1])

        return x, y

    @staticmethod
    def extreme_points(points_list):

        y, x = [], []

        for point in points_list:
            x.append(point[0])
            y.append(point[1])

        min_x = min(x)
        max_y = max(y)

        p = (min_x, max_y)

        return p


    @staticmethod
    def new_points(c_r1, c_r2):
        new_c = [c_r1[0]]
        all_c = c_r1 + c_r2
        for point in all_c:
            flag = False
            if point in c_r1 and point not in c_r2:
                flag = True
            if point in c_r2 and point not in c_r1:
                flag = True
            if flag is True and point not in new_c:
                new_c.append(point)

        return new_c

    @staticmethod
    def same_x(point, list_p, pp):

        # if point is in line with min x, go down
        if point[0] == pp[0]:
            try:
                p1 = next(x for x in list_p if x[0] == point[0] and x[1] < point[1])
                return p1
            except StopIteration:
                return False
        # else go up
        else:
            try:
                p1 = next(x for x in list_p if x[0] == point[0] and x[1] != point[1])
                return p1
            except StopIteration:
                return False

    @staticmethod
    def same_y(point, list_p, pp):

        # if point is in line with max y, go left
        if point[1] == pp[1]:
            try:
                p1 = next(x for x in list_p if x[1] == point[1] and x[0] < point[0])
                return p1
            except StopIteration:
                return False
        # else go right
        else:
            try:
                p1 = next(x for x in list_p if x[1] == point[1] and x[0] != point[0])
                return p1

            except StopIteration:
                return False






















