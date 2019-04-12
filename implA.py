import numpy as np
import sys
import random
import pygame
import math
from pygame.locals import *
import time
import requests
import operator
import copy
import matplotlib.pyplot as plt
import csv
import collections
from geopy.geocoders import Nominatim


# Rotate vector (poistive counterclockwise)
def rotate_vector(vector, angle):
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    vector = np.matmul(r, vector)
    return vector

class Weights:
    def __init__(self,layers):
        self.total = np.zeros((len(graph.nodes)))
        self.noise = np.random.rand(len(graph.nodes))*4
        self.weather = np.random.rand(len(graph.nodes))*4
        self.density =  np.ones((len(graph.nodes)))
        self.temperature =  np.zeros((len(graph.nodes)))

        for node in graph.nodes:
            pt = graph.ids[node]
            r = math.sqrt(pt[0]**2+pt[1]**2+pt[2]**2)
            lat = (math.acos(pt[2]/r))*180/math.pi
            lon = (math.atan(pt[1]/pt[0]))*280/math.pi
            #print(lat)

            if lat > 90:
                lat -=180
            if lat <0 and lon <0:

                self.density[node] = 13

        self.layers = layers

        self.all = {'noise':self.noise,'weather':self.weather,'density':self.density,'temperature':self.temperature}

    def set_total(self):
        self.total = np.zeros((len(graph.nodes)))
        for layer in self.layers:
            self.total += self.all[layer]

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
    link_id = str(sorted([from_node, to_node]))
    if link_id not in self.distances.keys():
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[link_id] = distance

def Astar(graph,initial):
    open = {initial:0}
    closed = {}

    nodes = set(graph.nodes)

    while nodes:


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

def fibonacci_sphere(samples=1.,randomize=True):
    start = time.time()
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(int(samples)):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append(np.array([x,y,z])*400000)
    return points

def set_edges():
    graph.edges = collections.defaultdict(list)
    graph.distances = {}
    for node in graph.nodes:
        for linked_node in graph.nodes:
            if linked_node != node:
                distance_l = np.linalg.norm(graph.ids[node]-graph.ids[linked_node])
                ang = math.acos((2*400000**2-distance_l**2)/(2*400000**2))

                w_distance = 400000*ang
                if  w_distance<3*distance:
                    w_tot = w_distance * (weights.total[node]+weights.total[linked_node])/2
                    graph.add_edge(node, linked_node, w_tot)



n = 30
m = 1

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

A = np.pi * 4 * 400000 ** 2

button_draw = pygame.Rect(50, 50, 50, 50)
button_weather = pygame.Rect(150, 50, 50, 50)
button_noise = pygame.Rect(250, 50, 50, 50)

# Game loop
while True:
    n = 30
    m = 1

    graph = Graph()
    n_points = 300
    distance = math.sqrt(A / n_points)

    fibsphere = fibonacci_sphere(n_points)
    for point in fibsphere:
        tr =  int((point[2]+400000)/800000*255)
       # pygame.draw.circle(screen, ((tr,tr,tr)), cartesian_to_screen(point), 4)


    pygame.display.flip()
    for point in fibsphere:
        size = int(len(graph.nodes))
        graph.add_node(size, point)




    weights = Weights(['density'])
    weights.set_total()


    print(graph.nodes)
    print(graph.ids)

    set_edges()
    print('--')
    print(graph.edges[0])
    center_node = random.randint(0, len(graph.nodes)-1)
    nxt = random.randint(0, len(graph.nodes)-1)

    last_point = graph.ids[nxt]
    center_point = graph.ids[center_node]
    distance_l = np.linalg.norm(last_point - center_point)
    ang = math.acos((2 * 400000 ** 2 - distance_l ** 2) / (2 * 400000 ** 2))

    mindist = 400000 * ang
    print('Min distance:',mindist)

    for i in range(4):
        # Fill black screen
        screen.fill((0, 0, 0))


        allpoints = []

        newpoits = fibonacci_sphere(n_points)
        distance = math.sqrt(A/n_points)
        if i ==0:
            path = graph.nodes
        else:
            for p in path:
                allpoints.append(graph.ids[p])

        start = time.time()
        for point in newpoits:
            for p in path:
                if np.linalg.norm(point-graph.ids[p])<distance*4:
                    allpoints.append(point)
        print('Time path', time.time()-start)
        allpoints.append(last_point)
        allpoints.append(center_point)
        allpoints =np.unique(np.array(allpoints),axis=0)



        graph = Graph()
        for point in allpoints:
                # Add node
                size = int(len(graph.nodes))

                if list(point) == list(center_point):
                    #print('redefinced center')
                    center_node = size
                if list(point) == list(last_point):
                    #print('redefined last')
                    nxt = size

                graph.add_node(size, point)
        weights = Weights(['density'])
        weights.set_total()

        start = time.time()
        set_edges()
        print('Time edges',time.time()-start)
        pygame.draw.circle(screen, red, cartesian_to_screen(last_point), 5)
        pygame.draw.circle(screen, red, cartesian_to_screen(center_point), 5)
        start = time.time()

        table = dijsktra(graph, center_node)
        print('Time dijsktra',time.time()-start)

        path = [nxt]
        while nxt != center_node:
            nxt = table[1][nxt]
            path.append(nxt)
        path = path[::-1]
        cost = 0
        for i in range(len(path) - 1):
            link_id = str(sorted([path[i], path[i+1]]))
            cost+= graph.distances[link_id]
            pygame.draw.line(screen, green, cartesian_to_screen(graph.ids[path[i]]),
                             cartesian_to_screen(graph.ids[path[i + 1]]), 2)
        print(cost,len(path))

        for node in graph.nodes:
            tr = int((graph.ids[node][2] + 600000) / 1200000 * 255)
            pygame.draw.circle(screen, ((tr, tr, tr)), cartesian_to_screen(graph.ids[node]),  int(weights.total[node]))
        for node in path:
            tr = int((graph.ids[node][2] + 600000) / 1200000 * 255)
            pygame.draw.circle(screen, yellow, cartesian_to_screen(graph.ids[node]),2)
        #pygame.display.flip()

        n_points *=4










        pygame.display.flip()




    pygame.draw.rect(screen, [255, 0, 0], button_draw)

    pygame.draw.rect(screen, [255, 0, 0], button_weather)

    pygame.draw.rect(screen, [255, 0, 0], button_noise)

    if len(drawing_pol)>1:
        for p in range(0,len(drawing_pol)-1):
            pygame.draw.line(screen, green, cartesian_to_screen(drawing_pol[p]),cartesian_to_screen(drawing_pol[p+1]), 1)

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
                        for node in graph.nodes:

                            weights.all[layer] += 0.2**(np.linalg.norm(graph.ids[node][:2]-screen_to_cartesian(mouse_pos)*1000)/distance)
                    weights.set_total()
                    set_edges()

                    table = dijsktra(graph, center_node)
                    print('once')


    # Draw screen
    pygame.display.flip()
    fpsClock.tick(fps)

# Close simulation
pygame.quit()
