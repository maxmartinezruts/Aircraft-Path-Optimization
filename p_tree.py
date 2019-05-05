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
    w = (weights[st.coor[0],st.coor[1],st.coor[2]]+weights[en.coor[0],en.coor[1], en.coor[2]])/2
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
        v3=[0.3,0]
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
# class Node:
#     def __init__(self):
#         self.predecessor = None
#         self.successor = None
#         self.pos = 0


class Point:
    def __init__(self, pos):
        self.pos = pos
        self.w =  abs(math.sin(pos[0])+math.sin(pos[1]))/(pos[2]+1)


class Graph:
    def __init__(self, i, f):
        self.path = [i,f]
        n = 100
        self.xs = np.linspace(-8,8,n)
        self.ys = np.linspace(-8,8,n)
        self.draw_nodes()

        for k in range(0, 1000):
            self.de_relax()
            self.try_new_pos()


    def draw_nodes(self):
        screen.fill((0, 0, 0))

        for x in self.xs:
            for y in self.ys:
                pos = np.array([x,y])
                w = abs(math.sin(pos[0])+math.sin(pos[1]))*10+1
                b = min(255, int(w * 100))
                pygame.draw.circle(screen, (b, 0, 0), cartesian_to_screen(np.array([x, y])), 3)
        pygame.display.flip()

    def try_new_pos(self):
        # Choose random node
        cs = []
        for p in range(1, len(self.path) - 1):
            path = list(self.path)
            path.pop(p)
            cost = self.get_path_cost(path)
            cs.append(cost)
        if min(cs) < self.get_path_cost(self.path):
            choice = cs.index(min(cs)) + 1
        else:

            choice = random.randint(1,len(self.path)-2)
        cs = []
        ps = []
        c1 = self.get_edge_cost(self.path[choice - 1].pos, self.path[choice].pos) + self.get_edge_cost(self.path[choice].pos, self.path[choice + 1].pos)
        for k in range(1000):
            delta = np.random.randn(3)
            delta[2]=0

            p = self.path[choice].pos + delta
            if 0<=p[2]<2:
                c =self.get_edge_cost(self.path[choice-1].pos, p) + self.get_edge_cost(p, self.path[choice+1].pos)
                cs.append(c)
                ps.append(p)
        pt = ps[cs.index(min(cs))]
        if list(pt) != list(self.path[choice].pos) and min(cs) < c1:
            self.path[choice] = Point(ps[cs.index(min(cs))])
            self.reconstruct_path(np.array([0,1,0], dtype=int))

    def de_relax(self):
        choice = random.randint(0,len(self.path)-2)
        cs = []
        ps = []
        c1 = self.get_edge_cost(self.path[choice].pos, self.path[choice + 1].pos)

        for k in range(1000):
            delta = np.random.randn(3)
            delta[2]=0

            p = self.path[choice].pos + delta
            if 0 <= p[2] < 2:
                c = self.get_edge_cost(self.path[choice].pos, p) + self.get_edge_cost(p, self.path[choice + 1].pos)
                cs.append(c)
                ps.append(p)
        pt = ps[cs.index(min(cs))]
        coll = False
        for p in self.path:
            if list(pt) == list(p.pos): coll = True
        if not coll and min(cs) < c1:
            self.path.insert(choice+1,Point(ps[cs.index(min(cs))]))
            self.reconstruct_path(np.array([1,0,0], dtype=int))


    def try_remove_worst(self):
        cs = []
        for p in range(1, len(self.path) - 1):
            path = list(self.path)
            path.pop(p)
            cost = self.get_path_cost(path)
            cs.append(cost)
        if min(cs) < self.get_path_cost(self.path):
            choice = cs.index(min(cs))+1

            self.path.pop(choice)
            print(min(cs), self.get_path_cost(self.path))
            self.reconstruct_path(np.array([1,1,1],dtype=int))


    def get_path_cost(self, path):
        cost = 0
        for p in range(len(path)-1):
            cost += self.get_edge_cost(path[p].pos, path[p + 1].pos)
        return cost

    def get_edge_cost(self, i, f):
        vec = f - i
        length = np.linalg.norm(vec)
        reps = int(length / 0.3)+2
        step = length / reps
        mults = np.linspace(0, 1, reps)
        cost = 0
        for j in range(reps-1):
            current = i + mults[j]*vec
            next =  i  + mults[j+1]*vec
            wi = abs(math.sin(current[0])+math.sin(current[1]))/(current[2]+0.01)
            wf = abs(math.sin(next[0])+math.sin(next[1]))/(next[2]+0.01)
            cost += (wi + wf)/2*step
        return cost


    def reconstruct_path(self, color):
        screen.fill((0, 0, 0))
        pygame.event.get()
        self.draw_nodes()
        for p in range(len(self.path)-1):
            c = tuple(color * int(min(255, 255 * self.path[p].pos[2])))
            c = tuple(color * 255)

            pygame.draw.line(screen, c, cartesian_to_screen(self.path[p].pos),cartesian_to_screen(self.path[p+1].pos), 3)
        print(self.get_path_cost(self.path), len(self.path))
        pygame.display.flip()
        # time.sleep(2)
        return self.path

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


graph = Graph(Point(np.array([-6,-4,0])), Point(np.array([4,4,0])))
pygame.quit()