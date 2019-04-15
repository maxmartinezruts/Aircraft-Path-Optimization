import numpy as np
import math
import random
import pygame
import time
from pprint import pprint
import collections


# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 0.021
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor-car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos


# Convert coordinates form pygame coordinates to screen coordinates
def screen_to_cartesian(screen_pos):
    car_pos = np.array([screen_pos[0]-center[0],center[1]-screen_pos[1]])
    car_pos = car_pos.astype(float)
    return car_pos

def heuristic_cost_estimate(st, en):
    rectilinear_distance = np.linalg.norm(st.car_pos - en.car_pos)
    ang = math.acos((2 * r ** 2 - rectilinear_distance ** 2) / (2 * r ** 2))
    circular_distance = r * ang
    return  circular_distance

def heuristic_cost_estimate_neighbour(st, en):
    w = (weights[st.coor[0],st.coor[1]]+weights[en.coor[0],en.coor[1]])/2
    rectilinear_distance = np.linalg.norm(st.car_pos - en.car_pos)
    ang = math.acos((2 * r ** 2 - rectilinear_distance ** 2) / (2 * r ** 2))
    circular_distance = r * ang
    return  circular_distance*w



def get_pos_by_coor(coor):
    return np.array([lonValues[coor[0]], latValues[coor[1]]])

def spherical_to_cartesian(pos):
    x = r*math.sin(pos[0])*math.cos(pos[1])
    y = r*math.sin(pos[0])*math.sin(pos[1])
    z = r*math.cos(pos[0])
    return np.array([x,y,z])


def get_coor_by_pos(pos):
    x =int(round((pos[0]-(0))/(1*math.pi-(0))*((n-1))))
    y =int(round((pos[1]-(0))/(2*math.pi-(0))*((n-1))))
    coor = np.array([x,y], dtype=int)

    return coor

class Node:
    def __init__(self, coor, parent):
        self.coor = coor
        self.pos =get_pos_by_coor(coor)
        self.car_pos = spherical_to_cartesian(get_pos_by_coor(coor))
        self.gScore = math.inf
        self.fScore = math.inf
        self.parent = None

    def get_neighbors(self):
        neightbours = []
        for lon in np.linspace(-4,4,20):
            for lat in np.linspace(-4,4,20):
                neighbourCoor=self.coor+np.array([lon,lat],dtype=int)
                if neighbourCoor[0] < 0:
                    neighbourCoor[0]+=n
                if neighbourCoor[0]>=n:
                    neighbourCoor[0]-=n
                if neighbourCoor[1] < 0:
                    neighbourCoor[1]+=n
                if neighbourCoor[1]>=n:
                    neighbourCoor[1]-=n
                if (graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]] == 0):
                    node = graph.add_node(neighbourCoor, self)
                else:
                    node = graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]]

                neightbours.append(node)
        return  neightbours


class Graph:
    def __init__(self):
        self.all_pos = []
        self.availableCoors = [[0 for j in range(n)] for i in range(n)]

    def prepare(self, start, end):
        print('new')
        self.start = start
        self.end = end
        # The set of nodes already evaluated
        self.closedSet = []

        # The set of currently discovered nodes that are not evaluated yet.
        #  Initially, only the start node is known.

        self.openSet = [start]

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.

    def search(self):
        current = self.start
        while len(self.openSet) > 0:  # While not close enough to end
            minScore = math.inf
            for node in self.openSet:
                if node.fScore < minScore:
                    minScore = node.fScore
                    current = node
            # current = the node in openSet having the lowest fScore[] value
            self.reconstruct_path(self.start, current,False,1)
            #print(current.pos)
            if np.linalg.norm(current.car_pos-end.car_pos) <0.2:
                print(current.parent)

                return self.reconstruct_path(self.start, current,True,3)

            self.openSet.remove(current)
            self.closedSet.append(current)
            neighbors = current.get_neighbors()

            for neighbor in neighbors:
                if neighbor in self.closedSet:
                    continue  # Ignore the neighbor which is already evaluated.

                # The distance from start to a neighbor
                tentative_gScore = current.gScore + heuristic_cost_estimate_neighbour(current, neighbor)

                if neighbor not in self.openSet:  # Discover a new node
                    self.openSet.append(neighbor)
                elif tentative_gScore >= neighbor.gScore:
                    continue
                # This path is the best until now. Record it!
                neighbor.parent = current
                neighbor.gScore = tentative_gScore
                neighbor.fScore = neighbor.gScore  + heuristic_cost_estimate(neighbor, self.end)

    def add_node(self, coor, parent):
        node = Node(coor,parent)
        self.availableCoors[coor[0]][coor[1]] = node
        self.all_pos.append(get_pos_by_coor(coor))

        return node
    def reconstruct_path(self, start, end, final, w):
        pygame.event.get()

        path = []
        current = end
        while current.parent != None:

            current = current.parent
            path.append(current)

        for p in range(len(path)-1):
            pos =spherical_to_cartesian(path[p].pos)
            height =int((((pos[2]+8)*255))/32)+40
            if not final:
                color = (height,height,height)
            else:
                color = yellow
            pygame.draw.line(screen, color, cartesian_to_screen(spherical_to_cartesian(path[p].pos)),cartesian_to_screen(spherical_to_cartesian(path[p+1].pos)), w)

        pygame.display.flip()

        return path

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

r = 8
fps = 40
n = 400
m =1000
# Construct grid
weights = np.ones((n,n))*1
# weights += np.random.rand(n,n)*2

mean_weights =  np.mean(weights)
print(np.mean(weights))
# weights = np.random.rand(100,100)*5
lonValues = np.linspace(0,math.pi,n)
latValues = np.linspace(0,math.pi*2,n)
xx,yy = np.meshgrid(latValues, lonValues)
for x in range(n):
    for y in range(n):
        weights[x,y] =1
        # weights[x,y] =abs(math.sin(get_pos_by_coor(np.array([0,y]))[1]))/+1
        # if 3/8*n < x < 5/8*n and 3/8*n < y < 5/8*n:
        #     weights[x, y] = 2

# Game loop
while True:
    screen.fill((0, 0, 0))
    #
    for x in range(n):
        for y in range(n):
            pygame.draw.circle(screen, red, cartesian_to_screen(spherical_to_cartesian(get_pos_by_coor([x, y]))), int(weights[x, y]*1))
    mean_weights = np.mean(weights)

    graph = Graph()
    stcor = np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int)
    start = graph.add_node(stcor, None)
    start.gScore = 0
    stcor = np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int)

    end = graph.add_node(stcor, None)
    pygame.draw.circle(screen, green, cartesian_to_screen(spherical_to_cartesian(start.pos)), 10)
    pygame.draw.circle(screen, yellow, cartesian_to_screen(spherical_to_cartesian(end.pos)), 10)
    graph.prepare(start, end)
    graph.search()

    # for node in graph.all:
    #     pygame.draw.circle(screen, yellow, cartesian_to_screen(node.pos), 1)

    print('smth')
    # Draw screen
    pygame.display.flip()
    time.sleep(1)
    fpsClock.tick(fps)

# Close simulation
pygame.quit()