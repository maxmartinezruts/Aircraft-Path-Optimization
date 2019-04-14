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
    return  np.linalg.norm(st.pos - en.pos)

def heuristic_cost_estimate_neighbour(st, en):
    w = (weights[st.coor[0],st.coor[1]]+weights[en.coor[0],en.coor[1]])/2
    return  np.linalg.norm(st.pos - en.pos)*w


def get_pos_by_coor(coor):
    return np.array([xValues[coor[0]], yValues[coor[1]]])

def get_coor_by_pos(pos):
    x =int(round((pos[0]-(-8))/(8-(-8))*((n-1))))
    y =int(round((pos[1]-(-8))/(8-(-8))*((n-1))))
    coor = np.array([x,y], dtype=int)

    return coor

class Node:
    def __init__(self, coor, parent):
        self.coor = coor
        self.pos = get_pos_by_coor(coor)
        self.gScore = math.inf
        self.fScore = math.inf
        self.parent = None

    def get_neighbors(self):

        if self.parent == None:
            angles = np.linspace(0,2*math.pi,400)
            v3 = np.array([0,0.3])
        else:
            angles =np.linspace(-0.2,0.2,20)
            v3 = self.pos - self.parent.pos

        neightbours = []

        for angle in angles:
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            v3rot = np.matmul(R,v3)
            neighbourPos = self.pos + v3rot

            neighbourCoor = get_coor_by_pos(neighbourPos)
            if 0 <neighbourCoor[0] <n and 0<neighbourCoor[1]<n:
                if ( graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]] == 0):
                    node =graph.add_node(neighbourCoor, self)
                else:
                    node = graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]]

                neightbours.append(node)
        return  neightbours


class Graph:
    def __init__(self):
        self.all = []
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
            self.reconstruct_path(self.start, current,green,1)
            #print(current.pos)
            if np.linalg.norm(current.pos - self.end.pos) <0.2:
                print(current.parent)

                return self.reconstruct_path(self.start, current,yellow,3)

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
        self.all.append(node)

        return node
    def reconstruct_path(self, start, end, color, w):
        pygame.event.get()

        path = []
        current = end
        while current.parent != None:

            current = current.parent
            path.append(current)

        for p in range(len(path)-1):
            pygame.draw.line(screen, color, cartesian_to_screen(path[p].pos),cartesian_to_screen(path[p+1].pos), w)

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


fps = 40
n = 150
m =10
# Construct grid
weights = np.ones((n,n))*1
# weights += np.random.rand(n,n)*2

mean_weights =  np.mean(weights)
print(np.mean(weights))
# weights = np.random.rand(100,100)*5
xValues = np.linspace(-8,8,n)
yValues = np.linspace(-8,8,n)
xx,yy = np.meshgrid(xValues, yValues)
for x in range(n):
    for y in range(n):
        weights[x,y] =abs(math.cos(get_pos_by_coor(np.array([x,0]))[0])+math.sin(get_pos_by_coor(np.array([0,y]))[1]))+1
        # if 3/8*n < x < 5/8*n and 3/8*n < y < 5/8*n:
        #     weights[x, y] = 2

# Game loop
while True:



    screen.fill((0, 0, 0))
    #
    for x in range(n):
        for y in range(n):
            pygame.draw.circle(screen, red, cartesian_to_screen(get_pos_by_coor([x, y])), int(weights[x, y]*2))
    mean_weights = np.mean(weights)

    graph = Graph()
    randvec = np.random.randint(n, size=2)
    stcor = np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int)
    start = graph.add_node(stcor, None)
    start.gScore = 0
    stcor = np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int)

    end = graph.add_node(stcor, None)
    pygame.draw.circle(screen, red, cartesian_to_screen(start.pos), 10)
    pygame.draw.circle(screen, yellow, cartesian_to_screen(end.pos), 10)
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
