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

class Weights:
    def __init__(self,n,layers):
        self.total = np.zeros((n,n, m))
        self.noise = np.random.rand(n,n,m)*9
        self.weather = np.random.rand(n,n,m)*4
        self.density =  np.zeros((n,n,m))
        for h in range(m):
            self.density[:,:,h] = m/(h+1)
        self.layers = layers
        self.all = {'noise':self.noise,'weather':self.weather,'density':self.density}

    def set_total(self):
        self.total = np.zeros((n, n, m))
        for layer in self.layers:
            self.total += self.all[layer]

class Graph:
  def __init__(self,n):
    self.nodes = set()
    self.edges = collections.defaultdict(list)
    self.distances = {}
    self.ids = {}
    self.coor_id = np.ones((n,n,m),dtype=int)
    print(self.coor_id[:,:,0])
    self.id_coor = {}


  def add_node(self, value, point,i,j,k):
    self.nodes.add(value)
    self.ids[value] = point
    self.coor_id[i,j,k] = value
    self.id_coor[value] = [i,j,k]

  def add_edge(self, from_node, to_node, distance):
    link_id = str(sorted([from_node, to_node]))
    if link_id not in self.distances.keys():
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[link_id] = distance
    else:
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[link_id] = distance



def dijsktra(graph, initial):
    print(graph,initial)
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

def set_edges():
    graph.edges = collections.defaultdict(list)
    graph.distances = {}
    for node in graph.nodes:
        i, j, k = graph.id_coor[node]
        for I in [-2, -1, 0, 1, 2]:
            for J in [-2, -1, 0, 1, 2]:
                for K in [-1, 0, 1]:

                    if 0 <= i + I < n and 0 <= j + J < n and 0 <= k + K < m and not (I == 0 and J == 0 and K == 0):
                        if graph.coor_id[i+I,j+J,k+K] !=-1:
                            w_distance = np.linalg.norm(
                                graph.ids[node] - graph.ids[graph.coor_id[i + I, j + J, k + K]]) / distance
                            w_tot = w_distance * ((weights.total[i, j, k] + weights.total[i + I, j + J, k + K]) + 0.01) / 2
                            #print(node, graph.coor_id[i+I, j+J,k+K],i+I,j+J)
                            graph.add_edge(node, graph.coor_id[i + I, j + J, k + K], w_tot)
n = 50
m = 1

edges = []
graph = Graph(n)


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

print(graph.nodes)
print(graph.ids)
path = []
x = np.linspace(-400000,400000,n)
y = np.linspace(-400000,400000,n)
z = np.linspace(0,10000,m)

weights = Weights(n,['density'])
weights.set_total()
distance = 8000000/(n-1)
for i in range(len(x)):
    for j in range(len(y)):
        for k in range(len(z)):
            if not(i<40 and 6<j<10):
                # Add node
                point = np.array([x[i], y[j], z[k]])*1000
                size = int(len(graph.nodes))
                graph.add_node(size, np.array([x[i], y[j], z[k]]),i,j,k)

print(graph.nodes)
print(graph.ids)
print(graph.coor_id[0,8,0])

set_edges()
print('--')
print(graph.edges[0])


pygame.display.flip()
time.sleep(1)

print(len(graph.distances))
print(graph.nodes)
print('-------------------')
center_node= graph.coor_id[int(n/2),int(n/2),0]
table = dijsktra(graph, center_node)

button_draw = pygame.Rect(50, 50, 50, 50)
button_weather = pygame.Rect(150, 50, 50, 50)
button_noise = pygame.Rect(250, 50, 50, 50)

# Game loop
while True:

    # Fill black screen
    screen.fill((0, 0, 0))

    for id in graph.nodes:
        i,j,k = graph.id_coor[id]
        pygame.draw.circle(screen, red, cartesian_to_screen(graph.ids[id]), int(weights.total[i,j,k]))



    nxt = random.randint(0, len(graph.nodes))
    closepoints = []
    path = [nxt]
    while nxt != center_node:
        nxt = table[1][nxt]
        path.append(nxt)
        for node in graph.nodes:
            #print(graph.ids[nxt]-graph.ids[node])
            if np.linalg.norm(graph.ids[nxt]-graph.ids[node])<(distance):
                closepoints.append(node)

        #print(graph.id_coor[nxt])
    print(len(set(closepoints)))
    closepoints = set(closepoints)

    path = path[::-1]
    for i in range(len(path)-1):
        pygame.draw.line(screen, yellow, cartesian_to_screen(graph.ids[path[i]]),
                         cartesian_to_screen(graph.ids[path[i+1]]), 1)


    pygame.draw.rect(screen, [255, 0, 0], button_draw)

    pygame.draw.rect(screen, [255, 0, 0], button_weather)

    pygame.draw.rect(screen, [255, 0, 0], button_noise)

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
            print(weights.layers)
            # If draw button clicked
            if button_noise.collidepoint(mouse_pos):
                if 'noise' in weights.layers:
                    weights.layers.remove('noise')
                else:
                    weights.layers.append('noise')
                weights.set_total()
                set_edges()
                table = dijsktra(graph, center_node)
            # If draw button clicked
            if button_weather.collidepoint(mouse_pos):
                if 'weather' in weights.layers:
                    weights.layers.remove('weather')
                else:
                    weights.layers.append('weather')
                weights.set_total()
                set_edges()
                table = dijsktra(graph, center_node)

            # If draw button clicked
            if button_draw.collidepoint(mouse_pos):

                # If drawing enabled, draw new polygon and disable drawing
                if drawing:
                    print('flsjfal')
                drawing = not drawing
            else:

                # If drawing enabled but mouse outside button
                if drawing:
                    print('clicked')
                    for layer in weights.layers:
                        for i in range(n):
                            for j in range(n):
                                for k in range(m):
                                    print(graph.coor_id[i,j,0])
                                    weights.all[layer][i,j,k] += 0.2**(np.linalg.norm(graph.ids[graph.coor_id[i,j,0]][:2]-screen_to_cartesian(mouse_pos)*1000)/distance)*3
                    weights.set_total()
                    set_edges()

                    table = dijsktra(graph, center_node)
                    print('once')


    # Draw screen
    pygame.display.flip()
    fpsClock.tick(fps)

# Close simulation
pygame.quit()
