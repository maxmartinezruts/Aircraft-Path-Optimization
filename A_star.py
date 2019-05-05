import numpy as np
import math
import random
import pygame
import time
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
            #+ abs(math.atan2(math.sin(st.ang-en.ang), math.cos(st.ang-en.ang)))/5

def heuristic_cost_estimate_neighbour(st, en):
    w = (weights[st.coor[0],st.coor[1]]+weights[en.coor[0],en.coor[1]])/2
    return  np.linalg.norm(st.pos - en.pos)*w


def get_pos_by_coor(coor):
    return np.array([xValues[coor[0]], yValues[coor[1]]])

def get_coor_by_pos(pos):
    x =int(round((pos[0]-(-8))/(8-(-8))*((n-1))))
    y =int(round((pos[1]-(-8))/(8-(-8))*((n-1))))
    k =int(round((pos[2]-(0))/(2*math.pi-(0))*((m-1))))
    coor = np.array([x,y,k], dtype=int)
    return coor

class Node:
    def __init__(self, state, parent):
        self.gScore = math.inf
        self.fScore = math.inf
        self.parent = parent
        self.state = state
        self.pos = state['pos']
        self.ang = state['b']
        self.coor = get_coor_by_pos(np.array([self.pos[0],self.pos[1],self.ang]))
        graph.availableCoors[self.coor[0]][self.coor[1]][self.coor[2]] = self

    def get_neighbors(self):


        neightbours = []
        v3=[1,0]
        R = np.array([[np.cos(self.ang), -np.sin(self.ang)], [np.sin(self.ang), np.cos(self.ang)]])
        v3rot = np.matmul(R, v3)
        neighbourPos = self.pos + v3rot
        for angle in kValues:

            if abs(math.atan2(math.sin(self.state['b']-angle), math.cos(self.state['b']-angle))) < 1 and -8 <neighbourPos[0] < 8 and -8 <neighbourPos[1] <8:
                neighbourCoor =  get_coor_by_pos(np.array([neighbourPos[0],neighbourPos[1], angle]))
                # neighbourPos = get_pos_by_coor(np.array([neighbourCoor[0],neighbourCoor[1]]))
                if (graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]][neighbourCoor[2]] == 0):
                    node = Node({'pos':neighbourPos, 'b':angle}, self)
                else:
                    node = graph.availableCoors[neighbourCoor[0]][neighbourCoor[1]][neighbourCoor[2]]
                neightbours.append(node)
        return  neightbours


class Graph:
    def __init__(self):
        self.all = []
        self.availableCoors = [[[0 for j in range(m)] for i in range(n)] for k in range(n)]

    def prepare(self, start, end):
        print('new')
        self.start = start
        self.end = end
        # The set of nodes already evaluated
        self.closedSet = []

        # The set of currently discovered nodes that are not evaluated yet.
        #  Initially, only the start node is known.

        self.openSet = [start]

        #         # For each node, which node it can most efficiently be reached from.
        #         # If a node can be reached from many nodes, cameFrom will eventually contain the
        #         # most efficient previous step.

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
            if np.linalg.norm(current.pos - self.end.pos) <0.3:
                    #and current.ang == self.end.ang:
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

    def add_node(self, state):
        node = Node(state)
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
n = 100
m =20
# Construct grid
weights = np.ones((n,n))*1
# weights += np.random.rand(n,n)*2

mean_weights =  np.mean(weights)
print(np.mean(weights))
# weights = np.random.rand(100,100)*5
xValues = np.linspace(-8, 8, n)
yValues = np.linspace(-8, 8, n)
kValues = np.linspace(0, 2*math.pi, m)
np.random.seed(1)
bumps = np.random.uniform(-6, 6, (30, 2))

def get_w(p):
    sigma = 0.3
    w=0
    for bump in bumps:
        w += 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (np.linalg.norm(bump-p))**2 / (2 * sigma**2))*5
    w +=1
    return w

for x in range(n):
    for y in range(n):

            weights[x,y] =get_w(get_pos_by_coor([x,y]))
            # weights[x, y, k] =1
# Game loop
while True:
    screen.fill((0, 0, 0))
    #
    for x in range(n):
        for y in range(n):

            brightness = min(255,int(weights[x, y]*100))
            pygame.draw.circle(screen, (brightness,0,0), cartesian_to_screen(get_pos_by_coor([x, y])), 3)
            pygame.draw.circle(screen, white, cartesian_to_screen(get_pos_by_coor([x, y])), 0)

    mean_weights = np.mean(weights)
    graph = Graph()
    randvec = np.random.randint(n, size=2)
    stpos = get_pos_by_coor(np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int))
    start = Node({'pos':np.array([-5,2]), 'b':math.pi}, None)
    start.gScore = 0
    stpos = get_pos_by_coor(np.array([random.randint(0,n-1),random.randint(0,n-1)],dtype=int))

    end = Node({'pos':np.array([4,4]), 'b':0}, None)
    pygame.draw.circle(screen, green, cartesian_to_screen(start.pos), 10)
    pygame.draw.circle(screen, yellow, cartesian_to_screen(end.pos), 10)
    pygame.display.flip()
    # time.sleep(10)

    graph.prepare(start, end)
    graph.search()

    # Draw screen

    pygame.display.flip()
    # time.sleep(20)
    fpsClock.tick(fps)

# Close simulation
pygame.quit()