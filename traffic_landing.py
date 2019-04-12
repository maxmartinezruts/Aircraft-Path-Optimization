import numpy as np
import sys
import random
import pygame
import math
from pygame.locals import *
import copy
import operator
import matplotlib.pyplot as plt
import numpy
import time

power = 2
lim = 20**(1/power)
A = np.zeros((50,50))

A[40,40] =1
for y in range(0,50):
    for x in range(0,50):
        if np.linalg.norm(np.array([x,y])-np.array([0,0]))<20:
            A[y,x] += -(lim*(np.linalg.norm(np.array([x,y])-np.array([0,0]))/20))**power + 20

for y in range(0,50):
    for x in range(0,50):
        if np.linalg.norm(np.array([x,y])-np.array([20,20]))<20:
            A[y,x] += -(lim*(np.linalg.norm(np.array([x,y])-np.array([20,20]))/20))**power + 20
for y in range(0,50):
    for x in range(0,50):
        if np.linalg.norm(np.array([x,y])-np.array([0,20]))<20:
            A[y,x] += -(lim*(np.linalg.norm(np.array([x,y])-np.array([0,20]))/20))**power + 20

plt.imshow(A, interpolation='none')
plt.show()
class Plane:
    def __init__(self, pos, size, t_gen, p_id):
        self.pos = pos
        self.size = size
        self.t_gen = t_gen
        self.id = p_id

    def tar_dist(self):
        return np.linalg.norm(self.pos)

    def t_min(self):
        return self.tar_dist()/ vel_max

    def get_coll_dist(self):
        dists = []
        for pl in planes.values():
            dists.append(np.linalg.norm(self.pos-pl.pos))
        return np.array(dists)

    def get_path(self):
        virtual_dist = float(self.virtual_dist)
        pos = np.array(self.pos)
        i = 0
        tmin = int(self.tar_dist()/self.vel)+2
        xs = np.arange(0, tmin)
        ys = np.arange(0, tmin)
        hor = np.ones(tmin)*100
        while virtual_dist > 0:
            # Position vectorization
            target_vector = -pos
            target_distance = np.linalg.norm(target_vector)
            target_direction = target_vector / target_distance

            # If velocity too low, change angle so that min velocity is fulfilled (spiral behaviour)
            if self.vel < vel_min:
                # Cosine rule
                a = np.linalg.norm(pos)
                b = np.linalg.norm(pos) - self.vel
                c = vel_min
                val = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
                if val > 1:
                    deviation = math.pi
                else:
                    deviation = math.acos(val)

                # Rotate direction of aircraft
                step = vel_min * rotate_vector(target_direction, deviation)

            # If velocity in bounds: OK
            elif vel_min <= self.vel <= vel_max:
                step = self.vel * target_direction

            # If valocity exceed max_vel, simply use max_vel
            elif vel_max < self.vel:
                step = vel_max * target_direction

            # Add step to pos
            pos += step
            virtual_dist -= self.vel

            pygame.draw.circle(screen, (40,40,40), cartesian_to_screen(pos), 1)
            xs[i]=pos[0]
            ys[i]=pos[1]
            i+=1
        print('xs',xs.shape)
        self.xs = xs
        self.ys = ys


def rotate_vector(vector, angle):
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    vector = np.matmul(r, vector)
    return vector


def cartesian_to_screen(car_pos):
    screen_pos = np.array([center[0]+car_pos[0],center[1]-car_pos[1]])
    screen_pos = screen_pos.astype(int)
    return screen_pos


def generate_plane(plane_id):
    global plane_count

    # Generate random position in polar coordinates
    alpha = random.uniform(0, 2*math.pi)

    #radius = random.uniform(rad_landing_1+500, rad_landing_1+600)
    radius = 500 + plane_id

    # Transform to cartesian and create position array
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    pos = np.array([x, y])

    # Generate plane
    tar_dist = np.linalg.norm(pos)
    t_min =  tar_dist / vel_max
    size = plane_id%3+1
    plane = Plane(pos, size, t, plane_id)
    plane.virtual_dist = plane.tar_dist()
    planes[plane_id] = plane

    plane_count += 1


def get_min(remaining, plane_id, sequence):
    sequence = copy.deepcopy(sequence)

    if len(sequence['order']) == 0:
        sequence['total_time'] = max(planes[plane_id].t_min(),separations[last_arrival_size][planes[plane_id].size])
        sequence['total_fuel'] = planes[plane_id].size * sequence['total_time']

    else:
        last = sequence['order'][-1]
        sequence['total_time'] = max(sequence['total_time']+separations[planes[last].size][planes[plane_id].size], planes[plane_id].t_min())
        sequence['total_fuel'] += planes[plane_id].size * sequence['total_time']

    # Update lists

    sequence['order'].append(plane_id)
    # Only way to copy a dictionary without the linked reference
    rem = copy.deepcopy(remaining)
    rem[planes[plane_id].size].remove(plane_id)

    # Delete an id from remaining list if the list of this size is already empty
    for size in list(rem):
        if len(rem[size]) == 0:
            rem.pop(size)

    # Check if sequence is finished
    if len(rem) == 0:
        return results.append(sequence)

    # In case there is an emergency landing but more planes can land in meanwhile

    #if 6 not in sequence and planes[6]['t_exp']>sequence['total_time']:
    #    get_min(rem, 6, sequence)

    for size in rem:
        # If differece of time generated airplane is small enough
        if len(emergencies)==0:
            constraint = 100
        else:
            constraint = 1000000
        #if (planes[plane_id].t_gen-planes[rem[size][0]].t_gen)<constraint:
        if (plane_id - rem[size][0]) < 20:

            get_min(rem, rem[size][0], sequence)


# Sequence finder for optimal fuel and time
def get_sequence():
    grouped_planes = {1: {}, 2: {}, 3: {}}
    first_seq = list(planes.keys())[:25]
    last_seq = list(planes.keys())[25:]
    for plane_id in first_seq:
        grouped_planes[planes[plane_id].size][plane_id] = planes[plane_id].tar_dist()
    si1 = list(dict(sorted(grouped_planes[1].items(), key=operator.itemgetter(1))))
    si2 = list(dict(sorted(grouped_planes[2].items(), key=operator.itemgetter(1))))
    si3 = list(dict(sorted(grouped_planes[3].items(), key=operator.itemgetter(1))))
    grouped_planes = {1: si1, 2: si2, 3: si3}
    print(grouped_planes)
    # Initialize
    print(planes.keys())
    if len(emergencies)>0:
        for emergency in emergencies:
            get_min(grouped_planes, emergency, {'order': [], 'total_time': 0, 'total_fuel': 0})
    else:
        for size in grouped_planes:
            if len(grouped_planes[size]) != 0:
                get_min(grouped_planes, grouped_planes[size][0], {'order': [], 'total_time': 0, 'total_fuel': 0})

    seq_fuels = sorted([x['total_fuel'] for x in results])
    seq_times = sorted([x['total_time'] for x in results])
    for seq in results:
        if seq_fuels[0] == seq['total_fuel']:
            sequences['fuel'] = list(seq['order'])+last_seq

        if seq_times[0] == seq['total_time']:
            sequences['time'] = list(seq['order'])+last_seq

    # Order by id ID
    sequences['id'] =list(planes.keys())

    # Order by distance to target
    distances = {}
    for plane_id in planes:
        distances[plane_id] = planes[plane_id].tar_dist()
    distances = sorted(distances.items(), key=operator.itemgetter(1))
    sequences['distance'] = []
    for dist in distances:
        sequences['distance'].append(dist[0])

    count = 0
    print(mode)
    for plane_id in sequences[mode]:

        if count == 0:
            previous_id = plane_id
            planes[plane_id].t_exp = max(planes[plane_id].t_min(),separations[last_arrival_size][planes[plane_id].size])
            planes[plane_id].vel = planes[plane_id].tar_dist() / planes[plane_id].t_exp

            hola =  planes[plane_id].t_exp
            adios =planes[plane_id].vel

        else:
            # Define time expected to target and velocity required
            planes[plane_id].t_exp = planes[previous_id].t_exp + separations[planes[previous_id].size][planes[plane_id].size]
            planes[plane_id].vel = planes[plane_id].tar_dist() / planes[plane_id].t_exp
            previous_id = plane_id
        count += 1


    for plane in planes.values():
        plane.get_path()

    for plane_id1 in planes:
        for plane_id2 in planes:
            if plane_id1 != plane_id2:
                xs1 = planes[plane_id1].xs
                xs2 = planes[plane_id2].xs
                min_l = min(len(xs1),len(xs2))-10
                min_l = min(min_l,100)
                xs1 = xs1[0:min_l]
                xs2 = xs2[0:min_l]
                ys1 = planes[plane_id1].ys[0:min_l]
                ys2 = planes[plane_id2].ys[0:min_l]

                idx_x = np.argwhere(abs(xs1-xs2)<40).flatten()
                idx_y = np.argwhere(abs(ys1-ys2)<40).flatten()
                idx = np.intersect1d(idx_x,idx_y)

                for i in idx:
                    point1 = np.array([planes[plane_id1].xs[i],planes[plane_id1].ys[i]])

                    pygame.draw.circle(screen, red, cartesian_to_screen(point1), 5)
    pygame.display.flip()
    time.sleep(0.4)
    return hola

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
separations = {
    1: {1: 82, 2: 69, 3: 60},
    2: {1: 131, 2: 69, 3: 60},
    3: {1: 196, 2: 157, 3: 66}
}

t = 0
t_last = 0
emergencies = [3]

# Planes
planes = {}
plane_count = 0
for i in range(0,12):
    generate_plane(plane_count)
last_arrival_size = 1

mode='time'
print(planes)
results = []
sequences= {}
get_sequence()



# Evaluate mission
fuel_total = 0
last_fuel = 0
time_aircrafts = 0
time_passangers = 0

landing = np.array([width/2, height/2])

pygame.init()

fps = 300
fpsClock = pygame.time.Clock()


fuel_axis = []
time_axis = []
show_path =  False


# Game loop.
while len(planes) > 0:
    time_axis.append(t)
    fuel_axis.append(fuel_total)
    t += 1

    screen.fill((0, 0, 0))
    # Draw elements
    button_emergency =  pygame.Rect(100, 100, 50, 50)
    button_path =  pygame.Rect(200, 200, 50, 50)
    button_mode =  pygame.Rect(300, 300, 50, 50)


    pygame.draw.rect(screen, [255, 0, 0], button_emergency)  # draw button
    pygame.draw.rect(screen, [255, 0, 0], button_path)  # draw button
    pygame.draw.rect(screen, [255, 0, 0], button_mode)  # draw button


    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos  # gets mouse position

            # checks if mouse position is over the button
            if button_emergency.collidepoint(mouse_pos):
                pl = random.sample(sequences[mode],1)[0]
                if pl not in emergencies:
                    print(pl)
                    emergencies.append(pl)
                # prints current location of mouse
                print('button was pressed at {0}'.format(mouse_pos))
            if button_path.collidepoint(mouse_pos):
                show_path =  not show_path
                mode = 'time'
            if button_mode.collidepoint(mouse_pos):
                mode = input("Type mode:")

    pygame.draw.circle(screen, red, cartesian_to_screen([0,0]), 5)

    # Update velocities in terms of next airplane expected time

    # Draw
    for plane in planes.values():
        fuel_total += plane.size

        if show_path:
            plane.get_path()
        # Getting data from planes dictionary

        # Position vectorization
        target_vector = -plane.pos
        target_distance = np.linalg.norm(target_vector)
        target_direction = target_vector/target_distance

        # If velocity too low, change angle so that min velocity is fulfilled (spiral behaviour)
        if plane.vel < vel_min:
            # Cosine rule
            a = plane.tar_dist()
            b = plane.tar_dist() - plane.vel
            c = vel_min
            val =(a**2 + c**2 - b**2 )/(2 * a * c)
            if val >1:
                deviation = math.pi
            else:
                deviation = math.acos(val)

            # Rotate direction of aircraft
            step = vel_min * rotate_vector(target_direction, deviation)

        # If velocity in bounds: OK
        elif vel_min <= plane.vel <= vel_max:
            step = plane.vel * target_direction

        # If valocity exceed max_vel, simply use max_vel
        elif vel_max < plane.vel:
            step = vel_max * target_direction

        # Add step to pos
        plane.pos += step
        plane.virtual_dist -= plane.vel

        # Draw airplane
        if plane.id in emergencies:
            color = red
        else:

            color = white
        pygame.draw.circle(screen, color, cartesian_to_screen(plane.pos), plane.size * 2)

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
            print('Time: ',t,'Total fuel: ',fuel_total, 'Mode:', mode)
            print('---------')
            t_last = t
            last_arrival_size = planes[plane_id].size
            del(planes[plane_id])
            generate_plane(plane_count)
            for seq_id in sequences:
                sequences[seq_id].remove(plane_id)
            results = []
            sequences = {}
            get_sequence()
    d_fuel = fuel_total- last_fuel
    #print(d_fuel)
    last_fuel = fuel_total

    pygame.display.flip()
    fpsClock.tick(fps)


    # Needed for drawing screen

pygame.quit()
print('fuel:', fuel_total, 'time:', t)
y = np.random.random()
x = t
plt.plot(time_axis,fuel_axis,'ro')
plt.show()