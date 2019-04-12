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
    def __init__(self,layers):
        self.total = np.zeros((len(graph.nodes)))
        self.noise = np.random.rand(len(graph.nodes))*4
        self.weather = np.random.rand(len(graph.nodes))*4
        self.density =  np.zeros((len(graph.nodes)))
        self.layers = layers

        self.all = {'noise':self.noise,'weather':self.weather,'density':self.density}

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

def fibonacci_sphere(samples=1,randomize=True):
    start = time.time()
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        print(r)
        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append(np.array([x,y,z])*400000)
    print(time.time()-start,'time spent')
    return points

def set_edges():
    graph.edges = collections.defaultdict(list)
    graph.distances = {}
    for node in graph.nodes:
        for linked_node in graph.nodes:
            w_distance = np.linalg.norm(graph.ids[node]-graph.ids[linked_node])/distance
            if linked_node != node and w_distance<2:
                w_tot = w_distance * ((weights.total[node] + weights.total[linked_node]) + 0.01) / 2
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



button_draw = pygame.Rect(50, 50, 50, 50)
button_weather = pygame.Rect(150, 50, 50, 50)
button_noise = pygame.Rect(250, 50, 50, 50)

# Game loop
while True:
    n = 30
    m = 1

    graph = Graph()
    fibsphere = fibonacci_sphere(400)
    for point in fibsphere:
        tr =  int((point[2]+400000)/800000*255)
       # pygame.draw.circle(screen, ((tr,tr,tr)), cartesian_to_screen(point), 4)


    pygame.display.flip()
    for point in fibsphere:
        size = int(len(graph.nodes))
        graph.add_node(size, point)



    distance = 100000

    weights = Weights(['density'])
    weights.set_total()


    print(graph.nodes)
    print(graph.ids)

    set_edges()
    print('--')
    print(graph.edges[0])
    center_node = random.randint(0, len(graph.nodes)-1)




    nxt = random.randint(0, len(graph.nodes)-1)
    for i in range(3):
        table = dijsktra(graph, center_node)
        # Fill black screen
        screen.fill((0, 0, 0))
        screen.fill((0, 0, 0))

        print(len(graph.distances),len(graph.nodes))
        last_point = graph.ids[nxt]
        center_point = graph.ids[center_node]
        closenodes = []

        closenodes.append(center_node)
        closenodes.append(nxt)


        path = [nxt]
        while nxt != center_node:
            closenodes.append(nxt)
            for node in graph.nodes:
                # print(graph.ids[nxt]-graph.ids[node])
                if np.linalg.norm(graph.ids[nxt] - graph.ids[node]) < (distance)*1:
                    closenodes.append(node)
            nxt = table[1][nxt]
            path.append(nxt)
            closenodes.append(nxt)
            for node in graph.nodes:
                # print(graph.ids[nxt]-graph.ids[node])
                if np.linalg.norm(graph.ids[nxt] - graph.ids[node]) < (distance)*1.5:
                    closenodes.append(node)
        for id in graph.nodes:
            tr = int((graph.ids[id][2] + 400000) / 800000 * 255)
            pygame.draw.circle(screen, ((tr, tr, tr)), cartesian_to_screen(graph.ids[id]), int(weights.total[id])+4)

        path = path[::-1]
        for i in range(len(path) - 1):
            pygame.draw.line(screen, green, cartesian_to_screen(graph.ids[path[i]]),
                             cartesian_to_screen(graph.ids[path[i + 1]]), 1)
        pygame.display.flip()
        time.sleep(1)
            #print(graph.id_coor[nxt])
        closenodes = set(closenodes)
        closepoints = []
        for node in closenodes:
            closepoints.append( graph.ids[node])
        allpoints = closepoints
        for node1 in closenodes:
            for node2 in closenodes:
                p1 =graph.ids[node1]
                p2 =graph.ids[node2]
                if node1 != node2  and np.linalg.norm(graph.ids[node1]-graph.ids[node2])/distance<1.2:

                    allpoints.append(p1+(p2-p1)/2)
        allpoints = np.array(allpoints)

        allpoints =np.unique(allpoints,axis=0)
        graph = Graph()
        for point in allpoints:
                # Add node
                size = int(len(graph.nodes))

                if list(point) == list(center_point):
                    print('redefinced center')
                    center_node = size
                if list(point) == list(last_point):
                    print('redefined last')
                    nxt = size

                graph.add_node(size, point)
        weights = Weights(['density'])
        weights.set_total()
        distance /=2
        set_edges()
        pygame.draw.circle(screen, red, cartesian_to_screen(last_point), 5)
        pygame.draw.circle(screen, red, cartesian_to_screen(center_point), 5)







    pygame.display.flip()




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
