import numpy as np
import sys
import random
import pygame
import math
from pygame.locals import *

import matplotlib.pyplot as plt
import time
a = np.interp(1.5,[1,2],[2,3])


class Point:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def create_circe(center, radius):
    n_points = 15
    points = []
    for i in range(n_points):
        ang = math.pi*2*i/n_points
        point =center+  np.array([math.cos(ang)*radius,math.sin(ang)*radius])
        points.append(point.tolist())
    return points

def create_square(center, width):
    points = [[-width,-width],[-width,width],[width,width],[width,-width]]
    for p in range(0,len(points)):
        points[p] =(center+np.array(points[p])).tolist()
        print(points[p])
    return points

class Polygon:
    def __init__(self, points):
        self.points = points
        self.nexts_clock = []
        self.nexts_counter = []
        i= 0
        points_dic = {}
        for point in points:
            points_dic[i] = point
            i+=1
        idx = np.arange(len(points))
        self.points_dic = points_dic
        self.clockwise = np.roll(idx, 1)
        self.counterclockwise = np.roll(idx, -1)

class Plane:
    def __init__(self, pos, size, t_gen, p_id):
        self.pos = pos
        self.size = size
        self.t_gen = t_gen
        self.id = p_id
        self.vel = vel_max
        self.get_path()


    class Path:
        def __init__(self, seq):
            vectors = []
            lengths = []
            times = []
            seq = seq[::-1]
            for p in range(len(seq)-1):
                vec = np.array(seq[p+1]) - np.array(seq[p])
                vectors.append(vec)
                lengths.append(np.linalg.norm(vec))

            self.points = seq
            self.vectors = np.array(vectors)
            self.lengths = np.array(lengths)
            self.cumlengths = np.cumsum(lengths)
            self.cumtimes = np.cumsum(lengths)/vel_max
            self.cumvectors = np.cumsum(vectors)

            self.lengthpath = self.cumlengths[-1]


    def tar_dist(self):
        return np.linalg.norm(self.pos)

    def t_min(self):
        return self.tar_dist()/ vel_max
    def get_coll_dist(self):
        dists = []
        for pl in planes.values():
            dists.append(np.linalg.norm(self.pos-pl.pos))
        b = np.array(dists)
        return b

    def next_point(self, pos, path):
        intersects = []
        print('-----',path[-1])

        for polygon in polygons:

            for p in polygon.points_dic:
                if polygon.points_dic[p] != path[-1] and polygon.points_dic[polygon.counterclockwise[p]] != path[-1] :
                    A = pos
                    B = [0, 0]
                    C = polygon.points_dic[p]
                    D = polygon.points_dic[polygon.counterclockwise[p]]

                    if intersect(Point(A), Point(B), Point(C), Point(D)):
                        if polygon not in intersects:
                            intersects.append(polygon)
                            print('p',polygon.points_dic[p],polygon.points_dic[polygon.counterclockwise[p]])
        min_ang = 7
        max_ang = -7
        checked = []
        v1 = np.array([0, 0]) - pos
        for i in intersects:
            print(i.points)
        if len(intersects)>0:

            while len(intersects)>0:

                for polygon in intersects:
                    checked.append(polygon)
                    for i in polygon.points_dic:
                        if polygon.points_dic[i] != path[-1]:
                            v2 = np.array(polygon.points_dic[i]) - pos
                            ang = get_angle(v1,v2)
                            if ang <= min_ang:
                                min_ang = ang
                                min_pt = polygon.points_dic[i]
                                min_pl = polygon
                            if ang >= max_ang:
                                max_pt = polygon.points_dic[i]
                                max_ang = ang
                                max_pl = polygon

                intersects = []
                for polygon in polygons:
                    if polygon not in checked:
                        for p in polygon.points_dic:
                            A = pos
                            P_MAX = max_pt
                            C = polygon.points_dic[p]
                            D = polygon.points_dic[polygon.counterclockwise[p]]
                            P_MIN = min_pt
                            if intersect(Point(A), Point(P_MIN), Point(C), Point(D)) or intersect(Point(A), Point(P_MAX), Point(C), Point(D)):
                                intersects.append(polygon)


            if max_pt not in path:
                pathcount = list(path)
                pathcount.append(max_pt)
                self.next_point(max_pt, pathcount)
            if min_pt not in path:
                pathclock = list(path)
                pathclock.append(min_pt)
                self.next_point(min_pt, pathclock)
        elif len(intersects) == 0:
            path.append([0, 0])
            path = self.Path(path)
            self.paths.append(path)

    def get_path(self):
        self.paths = []
        self.next_point(self.pos, [list(self.pos)])

        minlen = 10000000
        print(self.pos)

        for path in self.paths:
            print('a path is found', path.points)
            for p in range(1, len(path.points)):
                pygame.draw.line(screen, green, cartesian_to_screen(path.points[p - 1]),
                                 cartesian_to_screen(path.points[p]), 1)
            pygame.display.flip()
            screen.fill((0, 0, 0))
            if path.lengthpath < minlen:
                print('fsafasdfadf')
                minlen = path.lengthpath
                minpath = path
        print(minpath)
        self.minpath = minpath
        self.virtual_dist = self.minpath.lengthpath

    def get_pos(self):
        tot = np.array([0, 0])
        a = int(len(self.minpath.vectors))
        for i in range(0, a):

            if self.minpath.cumlengths[i] <= self.virtual_dist:

                tot = tot + self.minpath.vectors[i]
            else:
                tot = tot + self.minpath.vectors[i] / self.minpath.lengths[i] * (
                            self.minpath.lengths[i] - (self.minpath.cumlengths[i] - self.virtual_dist))
                break
        self.pos = tot
        return tot

def get_angle(vector1, vector2):

    dot = vector1[0] * vector2[0] + vector1[1] * vector2[1]  # dot product
    det = vector1[0] * vector2[1] - vector1[1] * vector2[0]  # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def squarer(point):
    pos = planes[0].pos
    tar = np.array([0, 0])
    return get_angle(pos-tar,pos-point)


def rotate_vector(vector, angle):
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    vector = np.matmul(r, vector)
    return vector


def cartesian_to_screen(car_pos):
    screen_pos = np.array([center[0]+car_pos[0],center[1]-car_pos[1]])
    screen_pos = screen_pos.astype(int)
    return screen_pos

def screen_to_cartesian(screen_pos):
    car_pos = np.array([screen_pos[0]-center[0],center[1]-screen_pos[1]])
    car_pos = car_pos.astype(int)
    return car_pos


def generate_plane(plane_id):
    global plane_count
    # Generate random position in polar coordinates
    alpha = random.uniform(0, 2*math.pi)
    radius = random.uniform(rad_landing_1+500, rad_landing_1+600)

    # Transform to cartesian and create position array
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    pos = np.array([x, y])

    # Generate plane
    tar_dist = np.linalg.norm(pos)
    t_min =  tar_dist / vel_max
    size = 1
    plane = Plane(pos, size, t, plane_id)


   # print(pos,'paths:',plane.paths[0].cumlengths[-1])
    planes[plane_id] = plane
    #planes[plane_id] = {'pos': pos, 'vel': vel_max, 'size': size, 'tar_dist':tar_dist, 't_exp': t_min, 't_gen':t}
    #print(planes[plane_id])
    plane_count += 1


# Screen
width = 800
height = 800
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)

# Airport dimensions
rad_landing_1 = 100
rad_landing_2 = 20

# Performance NOT CHANGE for pre defined aircrafts, since t_exp depends on vmax
vel_min = 1
vel_max = 2

# Separation times for aircraft vortexs (first level leading, second level trailing)

t = 0
t_last = 0
emergencies = [3]

# Polygons
polygons = []

polygons.append(Polygon(create_circe(np.array([100,200]),80)))


#polygons.append(Polygon(create_circe(np.array([100,-100]),70)))



#polygons.append(Polygon(create_circe(np.array([-200,-200]),80)))

#polygons.append(Polygon(create_circe(np.array([300,100]),20)))

#print(cir.points)

for polygon in polygons:
    angles = []
    v1 = np.array([1, 0])

    for i in polygon.points_dic:
        v2 = np.array(polygon.points_dic[i]) - np.array([0, 0])

        angles.append(get_angle(v1, v2))
    idx = angles.index(min(angles))
    point1 = polygon.points_dic[idx]
    idx2 = angles.index(max(angles))
    point2 = polygon.points_dic[idx2]
    polygon.nexts_clock.append(point2)
    polygon.nexts_counter.append(point1)
    pygame.draw.circle(screen, red, cartesian_to_screen(point1), 5)
    pygame.draw.circle(screen, red, cartesian_to_screen(point2), 5)

# Planes
planes = {}
plane_count = 0
for i in range(0,100):
    generate_plane(plane_count)
last_arrival_size = 1

landing = np.array([width/2, height/2])

pygame.init()

fps = 100
fpsClock = pygame.time.Clock()


show_path =  False
drawing = False
drawing_pol = []
pygame.display.flip()


# Game loop.
while len(planes) > 0:


    t += 1

    screen.fill((0, 0, 0))

    # Draw elements
    button_path =  pygame.Rect(200, 200, 50, 50)
    button_draw =  pygame.Rect(400, 200, 50, 50)


    pygame.draw.rect(screen, [255, 0, 0], button_path)  # draw button
    pygame.draw.rect(screen, [255, 0, 0], button_draw)  # draw button
    for polygon in polygons:
        for p in range(0,len(polygon.points)):
            pygame.draw.line(screen, (100, 100, 100), cartesian_to_screen(polygon.points_dic[p]),cartesian_to_screen(polygon.points_dic[polygon.counterclockwise[p]]), 1)
    if len(drawing_pol)>1:
        for p in range(0,len(drawing_pol)-1):
            pygame.draw.line(screen, (200, 200, 200), cartesian_to_screen(drawing_pol[p]),cartesian_to_screen(drawing_pol[p+1]), 3)


    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos  # gets mouse position


            if button_path.collidepoint(mouse_pos):
                show_path =  not show_path

            if button_draw.collidepoint(mouse_pos):
                if drawing:
                    polygons.append(Polygon(drawing_pol))
                    drawing_pol = []
                    for plane in planes.values():
                        plane.get_path()

                drawing = not drawing
            else:
                if drawing:
                    print(screen_to_cartesian(mouse_pos))
                    drawing_pol.append(list(screen_to_cartesian(mouse_pos)))
    pygame.draw.circle(screen, red, cartesian_to_screen([0,0]), 5)

    # Update velocities in terms of next airplane expected time

    # Draw
    for plane in planes.values():




        pos = plane.get_pos()
        pygame.draw.circle(screen, red, cartesian_to_screen(pos), plane.size * 4)

        plane.virtual_dist -= vel_max


        #pygame.draw.circle(screen, white, cartesian_to_screen(plane.pos), plane.size * 2)

    # Update distances and react for planes that arrived to target
    plane_ids = list(planes.keys())
    for plane_id in plane_ids:
        # Update distance to target

        # In case airplane arrived to target
        if planes[plane_id].virtual_dist -planes[plane_id].vel < 0.0001:
            if plane_id in emergencies:
                emergencies.remove(plane_id)

            #fuel_total
            print(t-t_last,'<- Time interval')
            #print('Time: ',t)
            print('---------')
            t_last = t
            last_arrival_size = planes[plane_id].size
            del(planes[plane_id])
            generate_plane(plane_count)

    pygame.display.flip()
    fpsClock.tick(fps)


    # Needed for drawing screen

pygame.quit()
