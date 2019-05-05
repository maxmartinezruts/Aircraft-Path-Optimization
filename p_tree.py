import numpy as np
import math
import pygame
import sys
import time

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_coor_by_pos(pos):
    x =int(round((pos[0]-(-8))/(8-(-8))*((n-1))))
    y =int(round((pos[1]-(-8))/(8-(-8))*((n-1))))
    if x < 0: x =0
    if x >= n: x = n-1
    if y < 0: y =0
    if y >= n: y = n-1
    coor = np.array([x,y], dtype=int)
    return coor
# Convert coordinates form cartesian to screen coordinates (used to draw in pygame screen)
def cartesian_to_screen(car_pos):
    factor = 0.021
    screen_pos = np.array([center[0]*factor+car_pos[0],center[1]*factor-car_pos[1]])/factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

def screen_to_cartesian(screen_pos):
    factor = 0.021
    car_pos = np.array([screen_pos[0]-center[0],center[1]-screen_pos[1]])*factor
    car_pos = car_pos.astype(float)
    return car_pos

def get_w(p):
    sigma = 0.3
    w=0
    for bump in bumps:
        w += 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (np.linalg.norm(bump-p))**2 / (2 * sigma**2))*5
    w +=1
    return w

class Point:
    def __init__(self, pos):
        self.pos = pos
        self.w =  abs(math.sin(pos[0])+math.sin(pos[1]))/(pos[2]+1)


class Graph:
    def __init__(self, p):
        self.path = p

    def search(self):
        self.de_relax()
        self.try_new_pos()
        # self.try_remove_worst()

    def try_new_pos(self):
        # Choose random node
        cps = []
        pps = []
        for choice in range(1, len(self.path) - 1):
            cs = []
            ps = []
            if choice == 1: q = self.path[0].pos + np.array([0.1,0,0])
            else : q =self.path[choice-2].pos
            for k in range(100):
                delta = np.random.randn(3)
                delta[2] /=10

                p = self.path[choice].pos + delta

                if 0 <= p[2] :
                    c = self.get_edge_cost(self.path[choice - 1].pos, p) + self.get_edge_cost(p,self.path[choice + 1].pos)
                    cs.append(c)
                    ps.append(p)
            if len(cs) > 0:
                pt = ps[cs.index(min(cs))]
                path = list(self.path)
                path[choice] = Point(pt)
                cps.append(self.get_path_cost(path))
                pps.append(path)
        if len(cps) > 0:
            self.path= pps[cps.index(min(cps))]
            self.reconstruct_path(np.array([0,1,0], dtype=int))

    def de_relax(self):
        cps = []
        pps = []
        for choice in range(0,len(self.path)-1):
            # choice = random.randint(0,len(self.path)-2)
            cs = []
            ps = []
            if choice == 0: q = self.path[0].pos + np.array([0.1,0,0])
            else : q =self.path[choice-1].pos

            for k in range(100):
                delta = np.random.randn(3)
                delta[2]/=10
                p = self.path[choice].pos + delta
                if 0 <= p[2]  :
                    c = self.get_edge_cost(self.path[choice].pos, p) + self.get_edge_cost(p, self.path[choice + 1].pos)
                    cs.append(c)
                    ps.append(p)

            if len(cs) > 0:

                path = list(self.path)
                path.insert(choice+1,Point(ps[cs.index(min(cs))]))
                cps.append(self.get_path_cost(path))
                pps.append(path)
        if len(cps) > 0:
            self.path= pps[cps.index(min(cps))]
            self.reconstruct_path(np.array([1,0,0], dtype=int))



    def try_remove_worst(self):
        cs = []
        for p in range(1, len(self.path) - 1):
            path = list(self.path)
            path.pop(p)
            cost = self.get_path_cost(path)
            cs.append(cost)
        if len(cs) > 0:
            if min(cs) < self.get_path_cost(self.path):
                choice = cs.index(min(cs))+1
                self.path.pop(choice)
                print(min(cs), self.get_path_cost(self.path))
                self.reconstruct_path(np.array([1,1,1],dtype=int))


    def get_path_cost(self, path):
        cost = 0
        for p in range(0,len(path)-1):
            if p ==0:
                q = path[0].pos + np.array([0.1,0,0])
            else:
                q = path[p-1].pos
            cost += self.get_edge_cost(path[p].pos, path[p + 1].pos)
        return cost

    def get_edge_cost(self, i, f):
        vec = f - i
        length = np.linalg.norm(vec)
        reps = int(length / 0.5)+2
        mults = np.linspace(0, 1, reps)
        cost = 0
        for j in range(reps-1):
            current = i + mults[j]*vec
            next =  i  + mults[j+1]*vec
            wi = weights[get_coor_by_pos(current)[0],get_coor_by_pos(current)[1]]/(current[2]+1)
            wf = weights[get_coor_by_pos(next)[0],get_coor_by_pos(next)[1]]/(next[2]+1)
            cost += (wi + wf)/2*np.linalg.norm(current-next)


        return cost


    def reconstruct_path(self, color):
        pygame.event.get()
        screen.fill((0, 0, 0))
        image = pygame.image.load('geek.jpg')
        screen.blit(image, (0, 0))

        for p in range(len(self.path)-1):
            c = tuple(color * int(min(255,  100 * self.path[p].pos[2])))
            pygame.draw.line(screen, c, cartesian_to_screen(self.path[p].pos),cartesian_to_screen(self.path[p+1].pos), 3)
            pygame.draw.circle(screen, white, cartesian_to_screen(self.path[p].pos), 3)

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
n = 100
xs = np.linspace(-8,8,n)
ys = np.linspace(-8,8,n)
np.random.seed(1)
bumps = np.random.uniform(-6, 6, (30, 2))
weights = np.zeros((n,n))



screen.fill((0, 0, 0))
for i in range(len(xs)):
    for j in range(len(ys)):
        w = get_w(np.array([xs[i],ys[j]]))
        weights[i,j] = w
        b = min(255, int(w * 100))
        pygame.draw.circle(screen, (b, 0, 0), cartesian_to_screen(np.array([xs[i], ys[j]])), 3)
pygame.image.save(screen, "geek.jpg")
screen.fill((0, 0, 0))
image = pygame.image.load('geek.jpg')
screen.blit(image, (0, 0))
pygame.display.flip()

waiting = True
graphs = []
path = []
while waiting:
    for p in path:
        pygame.draw.circle(screen, white, cartesian_to_screen(np.array([p.pos[0], p.pos[1]])), 3)
    pygame.display.flip()
    # Detect events in game

    for event in pygame.event.get():
        # When click event
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # Start path
                path = [Point(np.array([-5,2,0])), Point(np.array([4,4,0]))]
            if event.key == pygame.K_RIGHT:

                graphs.append(Graph(path))
            if event.key == pygame.K_SPACE:
                waiting = False
                # Create graph
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Record mouse position
            mouse_pos = event.pos
            pos = screen_to_cartesian(mouse_pos)
            pos = np.array([pos[0], pos[1], 0])
            print(pos)
            path.insert(len(path) - 1, Point(pos))
print('START    ')
while True:
    for graph in graphs:
        graph.search()

pygame.quit()