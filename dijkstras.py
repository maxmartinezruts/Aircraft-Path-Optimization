import numpy as np
import sys
import random
import pygame
import math
from pygame.locals import *
import time
import operator
import copy
import matplotlib.pyplot as plt
import csv
import collections

# Rotate vector (poistive counterclockwise)
def rotate_vector(vector, angle):
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    vector = np.matmul(r, vector)
    return vector

# Get curve points between two lines with required direction at target point
def get_curve(p_in, p_tar, v_out, rad):

    # Vector Initial-Final position
    v_tar = p_tar - p_in
    print(v_tar,v_out,'vsfas')
    # Angle between required direction and v_tar
    a_tar = get_angle(v_out, v_tar)

    # Specify direction and center of turn required
    if a_tar >0:
        turn ='clock'
        center = rotate_vector(v_out, -math.pi / 2) * rad + p_tar
    else:
        turn = 'counterclock'
        center = rotate_vector(v_out, math.pi / 2) * rad + p_tar

    # Vector Center-Initial
    center_out = p_in - center

    # All new points of curve will be stored here
    points = []

    # If not already inside circle of turn
    if np.linalg.norm(center_out) > rad:

        # Turn clockwise
        if turn == 'clock':
            # Interior angle of turn
            alpha = math.acos(rad/ np.linalg.norm(center_out))
            turn_point =  center + rotate_vector(center_out, -alpha)/np.linalg.norm(center_out)*rad
            angle = get_angle(p_tar-center,turn_point-center)
            # If negative, add one iteration
            if angle < 0: angle += 2 * math.pi
            # Add points
            while angle > 0:
                new_point = center + rotate_vector(p_tar-center, angle)
                points.append(list(new_point))
                angle -= 0.1
        else:                       # Turn counterclockwise
            alpha = math.acos(rad / np.linalg.norm(center_out))
            turn_point = center + rotate_vector(center_out, alpha) / np.linalg.norm(center_out) * rad
            angle = get_angle(turn_point - center, p_tar - center)

            if angle < 0: angle += 2 * math.pi
            while angle > 0:
                new_point = center + rotate_vector(p_tar - center, -angle)
                points.append(list(new_point))
                angle -= 0.1
        for point in points:
            pygame.draw.circle(screen, white, cartesian_to_screen(point), 2)
        screen.fill((0,0,0))
        # Draw results
        print(center,'centeeeeeer')
        pygame.draw.circle(screen, blue, cartesian_to_screen(center), 5)
        pygame.draw.circle(screen, green, cartesian_to_screen(turn_point), 5)

        for p in points:
            pygame.draw.circle(screen, white, cartesian_to_screen(p), 2)
        pygame.display.flip()

        #time.sleep(3)


    return points

class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = collections.defaultdict(list)
    self.distances = {}
    self.ids = {}

  def add_node(self, value, point):
    self.nodes.add(value)
    self.ids[value] = point

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[str(sorted([from_node, to_node]))] = distance



def dijsktra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes:
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distances[str(sorted([min_node, edge]))]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path

# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 1000
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor-car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos


# Convert coordinates form pygame coordinates to screen coordinates
def screen_to_cartesian(screen_pos):
    car_pos = np.array([screen_pos[0]-center[0],center[1]-screen_pos[1]])
    car_pos = car_pos.astype(float)
    return car_pos

# Easier acces coordinates
class Point:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

# Returns points are counterclockwise
def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# Get angle between 2 vectors (positive counterclockwise)
def get_angle(vector1, vector2):

    dot = vector1[0] * vector2[0] + vector1[1] * vector2[1]  # dot product
    det = vector1[0] * vector2[1] - vector1[1] * vector2[0]  # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle

class Polygon:
    def __init__(self, points):
        self.points = points
        self.nexts_clock = []
        self.nexts_counter = []
        self.node_ids = []
        i= 0
        points_dic = {}
        for point in points:
            print(type(point),'csafdfasga')
            points_dic[i] = point
            ln = len(graph.nodes)
            graph.add_node(ln,point)
            self.node_ids.append(ln)
            i+=1
        idx = np.arange(len(points))
        self.points_dic = points_dic
        print(self.points_dic)
        rotated = np.roll(idx, 1)
        for i in idx:
            graph.add_edge(self.node_ids[idx[i]],self.node_ids[rotated[i]], np.linalg.norm(points_dic[idx[i]]-points_dic[rotated[i]]))
            edges.append({'pts':np.array([points_dic[idx[i]],points_dic[rotated[i]]]),'pol':self, 'nodes':[self.node_ids[idx[i]],self.node_ids[rotated[i]]]})
        self.clockwise = np.roll(idx, 1)
        self.counterclockwise = np.roll(idx, -1)

def find_tangents(pol1, pol2):
    tangents = []
    checked = []
    for i in pol1.points_dic:
        for j in pol2.points_dic:

            a = pol1.points[i]
            b = pol2.points[j]
            # Find intersections(btn v and all other points)
            intersects = False
            for ed in edges:
                if ed['pol'] == pol1 or ed['pol'] == pol2:
                    A = a+(a-b)*50
                    B = b+(b-a)*50
                    C = ed['pts'][0]
                    D = ed['pts'][1]
                else:
                    A = a
                    B = b
                    C = ed['pts'][0]
                    D = ed['pts'][1]
                if not(np.any(ed['pts']==a) or np.any(ed['pts']==b)):

                    if  intersect(Point(A), Point(B), Point(C), Point(D)):
                        screen.fill((0, 0, 0))
                        pygame.draw.line(screen, yellow, cartesian_to_screen(A),
                                         cartesian_to_screen(B), 5)
                        pygame.draw.line(screen, blue, cartesian_to_screen(C),
                                         cartesian_to_screen(D), 5)
                        #time.sleep(0.2)
                        pygame.display.flip()
                        #if ed['pol'] == pol1  or ed['pol'] == pol2:
                        intersects = True
            if not intersects:
                print(checked)
                id = str(sorted([pol1.node_ids[i],pol2.node_ids[j]]))
                print(id)
                if not id in checked:
                    graph.add_edge(pol1.node_ids[i], pol2.node_ids[j], np.linalg.norm(a - b))
                    tangents.append([a, b])
                    checked.append(id)
                    # Found tangent


    return tangents
edges = []
graph = Graph()

# Screen parameters
width = 800
height = 800
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)

fpsClock = pygame.time.Clock()

# Polygons
polygons = []
drawing = False
drawing_pol = []
fps = 40
tangents = []

polygons.append(Polygon([np.array([0.0,0.0])]))
print(graph.nodes)
print(graph.ids)
path = []
# Game loop
while True:


    # Fill black screen
    screen.fill((0, 0, 0))

    button_draw =  pygame.Rect(50, 50, 50, 50)
    pygame.draw.rect(screen, [255, 0, 0], button_draw)
    for tangent in tangents:
        pygame.draw.line(screen, red, cartesian_to_screen(tangent[0]),
                         cartesian_to_screen(tangent[1]), 1)
    for edge in edges:
        pygame.draw.line(screen, white, cartesian_to_screen(edge['pts'][0]),
                         cartesian_to_screen(edge['pts'][1]), 2)

    for i in range(len(path)-1):
        pygame.draw.line(screen, yellow, cartesian_to_screen(graph.ids[path[i]]),
                         cartesian_to_screen(graph.ids[path[i+1]]), 3)
    # Time step


    if len(drawing_pol)>1:
        for p in range(0,len(drawing_pol)-1):
            pygame.draw.line(screen, green, cartesian_to_screen(drawing_pol[p]),cartesian_to_screen(drawing_pol[p+1]), 4)

    # Detect events in game
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # When click event
        if event.type == pygame.MOUSEBUTTONDOWN:

            # Record mouse position
            mouse_pos = event.pos


            # If draw button clicked
            if button_draw.collidepoint(mouse_pos):

                # If drawing enabled, draw new polygon and disable drawing
                if drawing:
                    polygons.append(Polygon(drawing_pol))

                    graph.edges = collections.defaultdict(list)
                    graph.distances = {}
                    print(graph.nodes)
                    for edge1 in edges:

                        print(edge1,edge1)
                        # If it doesn't intersect
                        intersection = False

                        for edge2 in edges:
                            if edge1['pol'] != edge2['pol']:
                                print('from other pol')
                                if intersect(Point(edge1['pts'][0]), Point(edge1['pts'][1]), Point(edge2['pts'][0]), Point(edge2['pts'][1])):
                                    intersection = True
                        if not intersection:
                            graph.add_edge(edge1['nodes'][0],edge1['nodes'][1], np.linalg.norm(edge1['pts'][0]-edge1['pts'][1]))
                    print(edges)
                    drawing_pol = []
                    tangents = []
                    for pol1 in polygons:
                        for pol2 in polygons:
                            if pol1 != pol2:
                                tangents += find_tangents(pol1,pol2)

                    print(graph.nodes,'edges')
                    print(dijsktra(graph,0))
                    table = dijsktra(graph,0)
                    nxt = 1
                    path = [nxt]
                    while nxt != 0:
                        nxt = table[1][nxt]
                        path.append(nxt)
                        print(nxt)
                path = path[::-1]
                if len(path) >1:
                    get_curve(graph.ids[path[0]],graph.ids[path[1]], graph.ids[path[2]]-graph.ids[path[1]],0.1)
                drawing = not drawing
            else:

                # If drawing enabled but mouse outside button
                if drawing:

                    # Append new point in polygon
                    drawing_pol.append(np.array((screen_to_cartesian(mouse_pos)*1000)))


    # Draw screen
    pygame.display.flip()
    fpsClock.tick(fps)

# Close simulation
pygame.quit()
