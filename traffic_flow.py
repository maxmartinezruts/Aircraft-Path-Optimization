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
from scipy import optimize



# Set secursion limit
sys.setrecursionlimit(10000)

# Initial number of planes
pl_ini = 40

class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = collections.defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[str(sorted([from_node, to_node]))] = distance

def get_time(seq):
    count = 0
    time_total = 0

    for plane_id in seq:
        # If count is 0 there is no previous plane. (leading plane)
        if count == 0:

            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[plane_id].t_min(),separations[last_arrival_size][planes[plane_id].size])

        else:
            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[previous_id].t_exp + separations[planes[previous_id].size][planes[plane_id].size], planes[plane_id].t_min())


        # Set previous to current for the following iteration
        previous_id = plane_id

        count += 1

        time_total += planes[plane_id].t_exp
        time_last = planes[plane_id].t_exp

    return time_total

# Get total fuel of sequence
def get_fuel(positions, converted=False):
    if not converted:
        all = np.arange(0,len(positions))
        seq_pos = {}
        for plane in all:
            seq_pos[plane] = positions[plane]
        seq = list(np.array(sorted(seq_pos.items(), key=lambda kv: kv[1]), dtype=int)[:,0])
    else:
        seq = positions
    count = 0
    fuel = 0

    for plane_id in seq:
        # If count is 0 there is no previous plane. (leading plane)
        if count == 0:

            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[plane_id].t_min(),separations[last_arrival_size][planes[plane_id].size])

        else:
            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[previous_id].t_exp + separations[planes[previous_id].size][planes[plane_id].size], planes[plane_id].t_min())

        # Set required velocity to land in time
        planes[plane_id].vel = planes[plane_id].tar_dist() / planes[plane_id].t_exp

        # Set privious to current for the following iteration
        previous_id = plane_id

        count += 1
        # Total fuel += time in air * fuel rate
        incr = planes[plane_id].t_exp*planes[plane_id].get_pow()

        fuel +=  incr
    return fuel

# Dynamic programming (minimize fuel)
def DP_swap_fuel(pl_seq,fuel_positions,pos,try_sequence):
    global sequence
   # print(sequence, pl_seq)
    # Plane id of relocating airplane
    pl_id = sequence[pl_seq]

    # Position of leading airplane
    pl_next = pos-1

    # Copy temporary sequence with no referencing
    try_sequence= copy.deepcopy(try_sequence)

    # If plane is leading
    if pos == 0:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos, pl_id)

        # Save fuel of this temporary sequence
        fuel_position = get_fuel(try_sequence, True)
        fuel_positions[pos] = fuel_position
        #print('pos  0', try_sequence, pl_seq, fuel_position)


        if len(fuel_positions) > 0:
            lst = list(fuel_positions.keys())
            new_pos = lst[list(fuel_positions.values()).index(min(fuel_positions.values()))]
        else:
            new_pos = pl_id

        # Relocate plane to most efficient position
        sequence.remove(pl_id)
        sequence.insert(new_pos, pl_id)

        # If not finished relocating all planes
        if pl_seq < len(sequence)-1:
            # Relocate next plane
            DP_swap_fuel(pl_seq + 1, {}, pl_seq + 1, sequence)

    # Once it encounters a plane of same size, it is known for sure that overpass this plane won't save any fuel
    elif planes[sequence[pl_next]].size == planes[pl_id].size:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos, pl_id)

        # Save fuel of this temporary sequence
        fuel_position = get_fuel(try_sequence,True)
        fuel_positions[pos] = fuel_position
        #print('same s', try_sequence, pl_seq, fuel_position)

        if len(fuel_positions)>0:
            lst = list(fuel_positions.keys())
            new_pos = lst[list(fuel_positions.values()).index(min(fuel_positions.values()))]
        else: new_pos = pl_id

        # Relocate plane to most efficient position
        sequence.remove(pl_id)
        sequence.insert(new_pos,pl_id)

        # If not finished relocating all planes
        if pl_seq < len(sequence)-1:
            # Relocate next plane
            DP_swap_fuel(pl_seq+1,{},pl_seq+1,sequence)

    # If plane not leading and next is not same size
    else:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos,pl_id)

        # Save fuel of this temporary sequence
        fuel_position = get_fuel(try_sequence, True)
        fuel_positions[pos] = fuel_position
        #print('clean ', try_sequence, pl_seq, fuel_position)

        # Try relocate plane one position aft
        DP_swap_fuel(pl_seq, fuel_positions, pos-1, try_sequence)


# Dynamic programming (minimize time)
def DP_swap_time(pl_seq,time_positions,pos,try_sequence):
    global sequence

    # Plane id of relocating airplane
    pl_id = sequence[pl_seq]

    # Position of leading airplane
    pl_next = pos-1

    # Copy temporary sequence with no referencing
    try_sequence= copy.deepcopy(try_sequence)

    # If plane is leading
    if pos == 0:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos, pl_id)

        # Save time of this temporary sequence
        time_position = get_time(try_sequence)
        time_positions[pos] = time_position

        if len(time_positions) > 0:
            lst = list(time_positions.keys())
            new_pos = lst[list(time_positions.values()).index(min(time_positions.values()))]
        else:
            new_pos = pl_id

        # Relocate plane to most efficient position
        sequence.remove(pl_id)
        sequence.insert(new_pos, pl_id)

        # If not finished relocating all planes
        if pl_seq < len(sequence)-1:
            # Relocate next plane
            DP_swap_time(pl_seq + 1, {}, pl_seq + 1, sequence)

    # Once it encounters a plane of same size, it is known for sure that overpass this plane won't save any time
    elif planes[sequence[pl_next]].size == planes[pl_id].size:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos, pl_id)

        # Save time of this temporary sequence
        time_position = get_time(try_sequence)
        time_positions[pos] = time_position

        if len(time_positions)>0:
            lst = list(time_positions.keys())
            new_pos = lst[list(time_positions.values()).index(min(time_positions.values()))]
        else: new_pos = pl_id

        # Relocate plane to most efficient position
        sequence.remove(pl_id)
        sequence.insert(new_pos,pl_id)

        # If not finished relocating all planes
        if pl_seq < len(sequence)-1:
            # Relocate next plane
            DP_swap_time(pl_seq+1,{},pl_seq+1,sequence)

    # If plane not leading and next is not same size
    else:

        # Temporary remove and add in first position
        try_sequence.remove(pl_id)
        try_sequence.insert(pos,pl_id)

        # Save time of this temporary sequence
        time_position = get_time(try_sequence)
        time_positions[pos] = time_position

        # Try relocate plane one position aft
        DP_swap_time(pl_seq, time_positions, pos-1, try_sequence)

# Initialize clock
start =time.time()


def DF_min(groups, plane_id, size):
    groups = copy.deepcopy(groups)
    groups[size].remove(plane_id)

    ln_1 =len(groups[1])
    ln_2 = len(groups[2])
    ln_3 =len(groups[3])
    l_rem = ln_1+ln_2+ln_3
    sizes = []
    if ln_1>0:
        sizes.append(1)
    if ln_2>0:
        sizes.append(2)
    if ln_3 > 0:
        sizes.append(3)
    if l_rem==0:
        return size*(4-l_rem)
    else:
        return min(DF_min(groups,groups[size][0],size) + size*(4-l_rem) for size in sizes)

# df = pd.read_csv('C:\\Users\\maxma\\PycharmProjects\\tf_test\\train_sequence.csv', sep=';', header=None)
# A = np.array(df.values, copy=True)  # Create new copy with no vinculation
# size_dataset = A.shape[0]
# limit_train =int(size_dataset*0.8)
# sizes_in = A[:,0:10]
# dists = A[:,10:20]/400000
# sizes_out = A[:,20:30]
# print(sizes_out[0])
# x = np.ones((size_dataset,10,3))
# for i in range(0,size_dataset):
#     for j in range(0,10):
#         x[i,j]= np.array([[sizes_in[i,j]].count(1.),[sizes_in[i,j]].count(2.),[sizes_in[i,j]].count(3.)])
# y = np.ones((size_dataset,10,3))
# for i in range(0,size_dataset):
#     for j in range(0,10):
#         y[i,j]= np.array([[sizes_out[i,j]].count(1.),[sizes_out[i,j]].count(2.),[sizes_out[i,j]].count(3.)])
#
# print(x[0])
# print(y[0])
# x_train = x[:limit_train]
# x_test = x[limit_train:]
#
# y_train = y[:limit_train]
# y_test = y[limit_train:]
#
# # Try replacing GRU, or SimpleRNN.
# RNN = layers.LSTM
# HIDDEN_SIZE = 128
# BATCH_SIZE = 128
# LAYERS = 1
#
# print('Build model...')
# model = Sequential()
# # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# # Note: In a situation where your input sequences have a variable length,
# # use input_shape=(None, num_feature).
# model.add(RNN(HIDDEN_SIZE, input_shape=(10,3)))
# # As the decoder RNN's input, repeatedly provide with the last output of
# # RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# # length of output, e.g., when DIGITS=3, max output is 999+999=1998.
# model.add(layers.RepeatVector(10))
# # The decoder RNN could be multiple layers stacked or a single layer.
# for _ in range(3):
#     # By setting return_sequences to True, return not only the last output but
#     # all the outputs so far in the form of (num_samples, timesteps,
#     # output_dim). This is necessary as TimeDistributed in the below expects
#     # the first dimension to be the timesteps.
#     model.add(RNN(HIDDEN_SIZE, return_sequences=True))
#
# # Apply a dense layer to the every temporal slice of an input. For each of step
# # of the output sequence, decide which character should be chosen.
# model.add(layers.TimeDistributed(layers.Dense(3, activation='softmax')))
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# model.summary()



# model.fit(x_train, y_train,batch_size=64, epochs=10,validation_data=(x_test, y_test))
# predictions = model.predict([x_test])
# print(predictions[0])
# for i in range(0,10):
#     print('Input: ',x_test[0][i], 'Output: ', y_test[0][i], 'Prediction: ', np.argmax(predictions[0][i])+1)
# print(x_test[0])
# print(y_test[0])
# print('--------------')

graph = Graph()
get_rho_by_vel = {}
specs = {1:{'Clmax':2, 'W':30000*9.81,  'S':72.7,'e':0.9, 'A':9.3, 'Cd0':0.025, 'Pavail':2*62300*70},2:{'Clmax':2, 'W':170000*9.81,'S':325,'e':0.9, 'A':10, 'Cd0':0.025, 'Pavail':60*1000000},3:{'Clmax':2, 'W':255000*9.81,'S':511,'e':0.9, 'A':6.97, 'Cd0':0.025, 'Pavail':150*1000000}}
for s in range(1,4):
    get_rho_by_vel[s] = {}

    CLmax = specs[s]['Clmax']
    W = specs[s]['W']
    S = specs[s]['S']
    e = specs[s]['e']
    A = specs[s]['A']
    Cd0 = specs[s]['Cd0']
    Pavail = specs[s]['Pavail']


    rho = 0.1

    for i in range(0,130):
        Preq = 0
        Vmax = (4 / 3 * (W / S) ** 2 / (rho ** 2 * Cd0 * math.pi * e * A)) ** (1 / 4)
        pows = []
        vels = []
        slopes = []
        while Pavail>Preq:
            Vmax+=1
            Preq = 0.5*rho*Vmax**3*S*Cd0 + 2*W**2/(rho*Vmax*S*math.pi*e*A)
            vels.append(Vmax)
            pows.append(Preq)
            slopes.append(Preq/Vmax)
        get_rho_by_vel[s][vels[slopes.index(min(slopes))]] = rho
        plt.plot(vels,pows)
        rho+=0.01
    plt.ylim(0)
    plt.xlim(0)

    #plt.show()
    list_vels = np.array(list(get_rho_by_vel[s].keys()))
    list_rhos = np.array(list(get_rho_by_vel[s].values()))
    for v in range(0,int(list_vels[0])):
        get_rho_by_vel[s][v] = list_rhos[np.abs(list_vels-v).argmin()]


h = 11000
a = -0.0065
rho0 = 1.225
T0 = 288
R = 287
g0 = 9.81
dt = 1
t_des = 600
V_des = 20
t=0
x=0
xs = []
ts = []
Vmaxs = []
Vmins = []
Vends = []
Vrans = []
hs = []
#plt.show()
while h>0:
    t+=dt
    h-= dt*V_des
    T = T0 + a * h
    rho=rho0 *(T/T0)**-((g0/(a*R))+1)
    Vend= (4 / 3 * (W / S) ** 2 / (rho ** 2 * Cd0 * math.pi * e * A)) ** (1 / 4)
    Vmin = math.sqrt(2 * W / (S * CLmax * rho))
    Preq = 0
    Vmax = Vend
    slopemin = 10000000000
    while Pavail > Preq:
        Vmax += 1
        Preq = 0.5 * rho * Vmax ** 3 * S * Cd0 + 2 * W ** 2 / (rho * Vmax * S * math.pi * e * A)
        slope = Preq/Vmax
        if slope<slopemin:
            slopemin = slope
            Vran = Vmax

    x+=Vend*dt

    xs.append(x)
    ts.append(t)
    hs.append(h)
    Vmaxs.append(Vmax)
    Vmins.append(Vmin)
    Vends.append(Vend)
    Vrans.append(Vran)

plt.plot(ts,xs)
#plt.show()

plt.plot(ts,hs)

#plt.show()

plt.plot(hs,Vmins)
plt.plot(hs,Vmaxs)
plt.plot(hs,Vends)
plt.plot(hs,Vrans)


#plt.show()

sizes = []
text = 'There are many variations a bit more of text of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour hello my name is max and this text is a random generator for the aircraft sizes. I am making it longer such that I can evaluate the performance of aircraft sequences for more time'
for ch in text:
    sizes.append(ord(ch)%3+1)


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

# Insert points given delay required
def insert_delay(p_in, p_tar, delay):
    # O delay (2*pi*r)
    angle = 2*math.pi
    points = []
    pygame.draw.circle(screen, red, cartesian_to_screen([0,0]), 5)

    while angle > 0:
        new_point = np.array([0,50]) + rotate_vector(np.array([50,0]) - np.array([0,0]), angle)
        points.append(list(new_point))
        pygame.draw.circle(screen, blue, cartesian_to_screen(new_point), 1)
        angle -= 0.1

    # C delay (pi/2*r + k + pi/2*r + k + pi/2*r + k + pi/2*r) - (4r - k)
    angle = math.pi/2
    while angle > 0:
        new_point = np.array([0,50]) + rotate_vector(np.array([0,-50]) - np.array([0,0]), angle)
        points.append(list(new_point))
        pygame.draw.circle(screen, green, cartesian_to_screen(new_point), 2)
        angle -= 10/50
    rect = 50
    while rect > abs(angle)*50:
        new_point = np.array([50, 50]) + np.array([0,1])* rect
        points.append(list(new_point))
        pygame.draw.circle(screen, red, cartesian_to_screen(new_point), 1)
        rect -= 10
    angle = math.pi/2
    while angle > abs(rect):
        new_point = np.array([100,100]) + rotate_vector(np.array([-50,0]), -angle)
        points.append(list(new_point))
        pygame.draw.circle(screen, green, cartesian_to_screen(new_point), 2)
        angle -= 10/50
    rect = 50
    while rect > abs(angle) * 50:
        new_point = np.array([100, 150]) + np.array([1, 0]) * rect
        points.append(list(new_point))
        pygame.draw.circle(screen, red, cartesian_to_screen(new_point), 1)
        rect -= 10
    angle = math.pi / 2
    while angle > abs(rect):
        new_point = np.array([150, 100]) + rotate_vector(np.array([0, 50]), -angle)
        points.append(list(new_point))
        pygame.draw.circle(screen, green, cartesian_to_screen(new_point), 2)
        angle -= 10 / 50
    rect = 50
    while rect > abs(angle) * 50:
        new_point = np.array([200, 100]) + np.array([0, -1]) * rect
        points.append(list(new_point))
        pygame.draw.circle(screen, red, cartesian_to_screen(new_point), 1)
        rect -= 10
    angle = math.pi / 2
    while angle > 0:
        new_point = np.array([250, 50]) + rotate_vector(np.array([-50, 0]) - np.array([0, 0]), angle)
        points.append(list(new_point))
        pygame.draw.circle(screen, green, cartesian_to_screen(new_point), 2)
        angle -= 10 / 50
    print(4*math.pi/2*50 + 3*50 - 5*50)

    # S delay
    angle = math.pi/4
    rad = 50
    k = 10
    i = 0
    while i < angle:
        new_point = np.array([0, 50]) + rotate_vector(np.array([0, -50]), i)
        points.append(list(new_point))
        pygame.draw.circle(screen, white, cartesian_to_screen(new_point), 3)
        i += 10 / 50
    last =np.array([0, 50]) + rotate_vector(np.array([0, -50]), angle)
    i=0
    while i < k:
        new_point = last + rotate_vector(np.array([1,0]), angle)*i
        points.append(list(new_point))
        pygame.draw.circle(screen, white, cartesian_to_screen(new_point), 3)
        i += 10
    last =last + rotate_vector(np.array([1,0]), angle)*k
    center =last + np.array([math.cos(angle)*50,-math.sin(angle)*50])
    i=0
    while i < (2*angle):
        new_point = center  + rotate_vector(last-center, -i)
        points.append(list(new_point))
        pygame.draw.circle(screen, white, cartesian_to_screen(new_point), 3)
        i += 10 / 50
    last = center + rotate_vector(last - center, -2*angle)
    i=0
    while i < k:
        new_point = last + rotate_vector(np.array([1,0]), -angle)*i
        points.append(list(new_point))
        pygame.draw.circle(screen, white, cartesian_to_screen(new_point), 3)
        i += 10
    last = last + rotate_vector(np.array([1, 0]), -angle) * k
    center = last + np.array([math.cos(angle) * 50, math.sin(angle) * 50])
    i = 0
    while i < ( angle):
        new_point = center + rotate_vector(last - center, i)
        points.append(list(new_point))
        pygame.draw.circle(screen, white, cartesian_to_screen(new_point), 3)
        i += 10 / 50

    pygame.display.flip()
    #time.sleep(0.1)

# Get curve points between two lines with required direction at target point
def get_curve(p_in, p_tar, v_out):
    # Vector Initial-Final position
    v_tar = p_tar - p_in

    # Angle between required direction and v_tar
    a_tar = get_angle(v_out, v_tar)

    # Specify direction and center of turn required
    if a_tar >0:
        turn ='clock'
        center = rotate_vector(v_out, -math.pi / 2) * rad_turn + p_tar
    else:
        turn = 'counterclock'
        center = rotate_vector(v_out, math.pi / 2) * rad_turn + p_tar

    # Vector Center-Initial
    center_out = p_in - center

    # All new points of curve will be stored here
    points = []

    # If not already inside circle of turn
    if np.linalg.norm(center_out) > rad_turn:

        # Turn clockwise
        if turn == 'clock':
            # Interior angle of turn
            alpha = math.acos(rad_turn/ np.linalg.norm(center_out))
            turn_point =  center + rotate_vector(center_out, -alpha)/np.linalg.norm(center_out)*rad_turn
            angle = get_angle(p_tar-center,turn_point-center)
            # If negative, add one iteration
            if angle < 0: angle += 2 * math.pi
            # Add points
            while angle > 0:
                new_point = center + rotate_vector(p_tar-center, angle)
                points.append(list(new_point))
                angle -= 0.1
        else:                       # Turn counterclockwise
            alpha = math.acos(rad_turn / np.linalg.norm(center_out))
            turn_point = center + rotate_vector(center_out, alpha) / np.linalg.norm(center_out) * rad_turn
            angle = get_angle(turn_point - center, p_tar - center)

            if angle < 0: angle += 2 * math.pi
            while angle > 0:
                new_point = center + rotate_vector(p_tar - center, -angle)
                points.append(list(new_point))
                angle -= 0.1
        for point in points:
            pygame.draw.circle(screen, white, cartesian_to_screen(point), 2)

        # Draw results
        pygame.draw.circle(screen, blue, cartesian_to_screen(center), 1)
        pygame.draw.circle(screen, green, cartesian_to_screen(turn_point), 1)
        pygame.display.flip()

    return points

# Create circle of points given center and radius
def create_circe(center, radius):
    n_points = 15
    points = []
    for i in range(n_points):
        ang = math.pi*2*i/n_points
        point =center + np.array([math.cos(ang)*radius,math.sin(ang)*radius])
        points.append(point.tolist())
    return points

# Create square of points given center and width
def create_square(center, width):
    points = [[-width,-width],[-width,width],[width,width],[width,-width]]
    for p in range(0,len(points)):
        points[p] =(center+np.array(points[p])).tolist()
        print(points[p])
    return points

# Rotate vector (poistive counterclockwise)
def rotate_vector(vector, angle):
    r = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    vector = np.matmul(r, vector)
    return vector

# Polygon class
class Polygon:
    def __init__(self, points):
        self.points = points
        self.nexts_clock = []
        self.nexts_counter = []
        self.node_ids = []
        i= 0
        points_dic = {}
        for point in points:
            points_dic[i] = point
            graph.add_node(len(graph.nodes))
            self.node_ids.append(len(graph.nodes))
            i+=1
        idx = np.arange(len(points))
        self.points_dic = points_dic
        print(self.points_dic)
        rotated = np.roll(idx, 1)
        for i in idx:
            graph.add_edge(self.node_ids[idx[i]],self.node_ids[rotated[i]], np.linalg.norm(points_dic[idx[i]]-points_dic[rotated[i]]))
        self.clockwise = np.roll(idx, 1)
        self.counterclockwise = np.roll(idx, -1)

class Species:
    def __init__(self, n_elements, mutation_rate, n_genes):
        self.n_generation = 0
        self.elements = []
        self.mutation_rate = mutation_rate
        self.n_elements = n_elements
        self.n_genes = n_genes
        self.scores = []
        for i in range(n_elements):
            self.elements.append(Sequence(n_genes))
    def pickOne(self):
        index = 0
        r = random.random()

        while r >0:
            r = r- self.elements[index].fitness
            index +=1
        index -=1

        element = Sequence(self.n_genes)
        element.pos_sequence = list(self.elements[index].pos_sequence)
        element.mutate(self.mutation_rate)
        return element


    def next_generation(self):
        for i in range(self.n_elements):
            self.elements[i].normalize()
        self.calculate_fitness()
        new_elements = []
        for i in range(self.n_elements):
            new_elements.append(self.pickOne())
        self.elements = new_elements


    def calculate_fitness(self):
        self.scores = []
        total = 0
        for i in range(self.n_elements):
            self.elements[i].get_score()
            self.scores.append(self.elements[i].score)
        max_score = max(self.scores)

        for i in range(self.n_elements):
            total += max_score - self.elements[i].score
        for i in range(self.n_elements):
            self.elements[i].get_fitness(total, max_score)
            # print(i, self.elements[i].fitness)

    def get_status(self):

        fuels = np.array(self.scores)
        mean = np.mean(fuels)
        min_fuel = np.min(fuels)

        return mean, min_fuel



class Sequence:
    def __init__(self, n_genes):
        self.score = 0
        self.fitness = 0
        self.pos_sequence = list(np.random.rand(n_genes))

    def mutate(self, mutation_rate):
        for pos in range(len(self.pos_sequence)):
            if random.random()<mutation_rate:
                self.pos_sequence[pos] += np.random.randn()
    def get_fitness(self, total, max_score):
        self.fitness = (max_score - self.score)/total
    def get_score(self):
        self.score = get_fuel(self.pos_sequence)
    def normalize(self):
        positions = []
        for i in range(len(self.pos_sequence)):
            positions.append(self.pos_sequence[i])
        positions = np.array(positions)
        min_pos = np.min(positions)
        max_pos = np.max(positions)
        # print(self.pos_sequence)
        for i in range(len(self.pos_sequence)):
            self.pos_sequence[i] = (self.pos_sequence[i]-min_pos)/(max_pos-min_pos)
        # print(self.pos_sequence)


class Plane:
    def __init__(self, pos, size, t_gen, p_id):
        self.pos = pos
        self.size = size
        self.t_gen = t_gen
        self.id = p_id
        self.vel = vel_max
        self.A = specs[self.size]['A']
        self.Pavail = specs[self.size]['Pavail']
        self.S = specs[self.size]['S']
        self.e = specs[self.size]['e']
        self.CLmax = specs[self.size]['Clmax']
        self.W = specs[self.size]['W']
        self.h = 11000
        self.a = -0.0065
        self.Cd0 =specs[self.size]['Cd0']
        self.get_path()
        self.rho = self.get_rho()
        self.Vran = self.get_Vran()

    class Path:
        def __init__(self, seq):
            vectors = []
            lengths = []

            for p in range(len(seq)-1):
                vec = np.array(seq[p+1]) - np.array(seq[p])
                vectors.append(vec)
                lengths.append(np.linalg.norm(vec))

            self.points = np.array(seq)
            self.vectors = np.array(vectors)
            self.lengths = np.array(lengths)
            self.cumlengths = np.cumsum(lengths)
            self.cumtimes = np.cumsum(lengths)/vel_max
            self.cumvectors = np.cumsum(vectors)
            self.lengthpath = self.cumlengths[-1]

    def get_pow(self):
        # Establish most efficient rho (altitude)
        self.rho = get_rho_by_vel[self.size][int(self.vel)]
        # Get power
        return 0.5 * self.rho * self.vel ** 3 * self.S * self.Cd0 + 2 * self.W ** 2 / (self.rho * self.vel * self.S * math.pi * self.e * self.A)

    def get_rho(self):
        T = T0 + self.a * self.h
        return rho0 * (T / T0) ** -((g0 / (self.a * R)) + 1)

    def Vmin(self):
        return math.sqrt(2 * self.W / (self.S * self.CLmax * self.rho))
    def Vmax(self):
        V = self.Vend()
        Preq = 0
        while self.Pavail > Preq:
            V += 1
            Preq = 0.5 * self.rho * V ** 3 * self.S * self.Cd0 + 2 * self.W ** 2 / (self.rho * V * self.S * math.pi * self.e * self.A)
        return V
    def Vend(self):
        return (4 / 3 * (self.W / self.S) ** 2 / (self.rho ** 2 * self.Cd0 * math.pi * self.e * self.A)) ** (1 / 4)
    def get_Vran(self):
        slopemin = 1000000000
        V = self.Vend()
        Preq = 0
        while self.Pavail > Preq:
            V += 1
            Preq = 0.5 * self.rho * V ** 3 * self.S * self.Cd0 + 2 * self.W ** 2 / (self.rho * V * self.S * math.pi * self.e * self.A)
            slope = Preq/V
            if slope < slopemin:
                slopemin = slope
                Vran = V
        return Vran

    # Returns distance required for land
    def tar_dist(self):
        return self.virtual_dist

    # Returns time required to land
    def t_min(self):
        return self.virtual_dist/ self.Vran

    # Returns distances to all aircraft
    def get_coll_dist(self):
        dists = []
        for pl in planes.values():
            dists.append(np.linalg.norm(self.pos-pl.pos))
        b = np.array(dists)
        return b

    # Find Collisions
    def next_point(self, pos, path):
        if len(path) == 1:
            d_out = np.array([-1, 0])
        else:
            vec = np.array(path[-2]) - np.array(path[-1])
            d_out = vec / np.linalg.norm(vec)
        for p in path:
            pygame.draw.circle(screen, yellow, cartesian_to_screen(p), 1)
        pygame.display.flip()
        #time.sleep(0.02)
        crv = get_curve(np.array(self.pos), pos, d_out)

        if len(crv)>0:

            pos_virt = np.array(crv[0])
        else:
            pos_virt = pos
        intersects = []

        for polygon in polygons:

            for p in polygon.points_dic:
                if (not(np.array_equal(polygon.points_dic[p],path[-1]))) and (not(np.array_equal(polygon.points_dic[polygon.counterclockwise[p]],path[-1]))) :
                    A = pos_virt
                    B = self.pos
                    C = polygon.points_dic[p]
                    D = polygon.points_dic[polygon.counterclockwise[p]]
                    print(A,B,C,D, type(A),type(D))

                    if intersect(Point(A), Point(B), Point(C), Point(D)):

                        if polygon not in intersects:
                            intersects.append(polygon)
        min_ang = 7
        max_ang = -7
        checked = []
        v1 = self.pos - pos_virt

        if len(intersects)>0:

            while len(intersects)>0:

                for polygon in intersects:
                    checked.append(polygon)
                    for i in polygon.points_dic:
                        if not np.array_equal(polygon.points_dic[i], path[-1]):
                            v2 = np.array(polygon.points_dic[i]) - pos_virt
                            ang = get_angle(v1,v2)
                            if ang <= min_ang:
                                min_ang = ang
                                min_pt = polygon.points_dic[i]
                            if ang >= max_ang:
                                max_pt = polygon.points_dic[i]
                                max_ang = ang
                # Checking collisions form current min/max positions to initial position
                intersects = []
                for polygon in polygons:
                    if polygon not in checked:
                        for p in polygon.points_dic:
                            A = pos
                            P_MAX = max_pt
                            C = polygon.points_dic[p]
                            D = polygon.points_dic[polygon.counterclockwise[p]]
                            P_MIN = min_pt
                            if intersect(Point(A), Point(P_MIN), Point(C), Point(D)) or intersect(Point(A), Point(P_MAX), Point(C), Point(D)):
                                intersects.append(polygon)

            if not np.any(path == max_pt):
                pathcount = list(path)
                crv = get_curve(np.array(max_pt),pos,d_out)
                for pt in crv[::-1]:
                    pathcount.append(list(pt))
                pathcount.append(max_pt)
                self.next_point(max_pt, pathcount)
            if not np.any(path == min_pt):
                crv = get_curve(np.array(min_pt),pos,d_out)
                pathclock = list(path)
                for pt in crv[::-1]:
                    pathclock.append(list(pt))
                pathclock.append(min_pt)

                self.next_point(min_pt, pathclock)

        elif len(intersects) == 0:

            crv = get_curve(np.array(self.pos), pos, d_out)
            for pt in crv[::-1]:
                    path.append(list(pt))
            path.append(self.pos)
            path = self.Path(path)
            self.paths.append(path)

    def get_path(self):
        self.paths = []
        self.next_point(np.array(np.array([0,0])), [np.array([0,0])])

        minlen = 10000000

        for path in self.paths:
            for p in range(1, len(path.points)):
                vec = np.array([path.points[p - 1],path.points[p]])
                if np.linalg.norm(vec)>300:
                    col = green
                else:
                    col = red
                pygame.draw.line(screen, col, cartesian_to_screen(path.points[p - 1]),
                                 cartesian_to_screen(path.points[p]), 1)
            #screen.fill((0, 0, 0))
            if path.lengthpath < minlen:
                minlen = path.lengthpath
                minpath = path
        self.minpath = minpath

        self.minpath = self.Path(self.minpath.points)

        self.virtual_dist = self.minpath.lengthpath
        self.xs = np.array(self.minpath.points)[:,0]
        self.ys = self.minpath.points[:,1]


    def get_pos(self):
        tot = np.array([0, 0])
        a = int(len(self.minpath.vectors))
        for i in range(0, a):

            if self.minpath.cumlengths[i] <= self.virtual_dist:

                tot = tot + self.minpath.vectors[i]
            else:
                tot = tot + self.minpath.vectors[i] / self.minpath.lengths[i] * (
                            self.minpath.lengths[i] - (self.minpath.cumlengths[i] - self.virtual_dist))
                break
        self.pos = tot
        return tot


# Get angle between 2 vectors (positive counterclockwise)
def get_angle(vector1, vector2):

    dot = vector1[0] * vector2[0] + vector1[1] * vector2[1]  # dot product
    det = vector1[0] * vector2[1] - vector1[1] * vector2[0]  # determinant
    angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def squarer(point):
    pos = planes[0].pos
    tar = np.array([0, 0])
    return get_angle(pos-tar,pos-point)


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


# Generate new plane
def generate_plane(plane_id):
    global plane_count
    # Generate random position in polar coordinates
    alpha = random.uniform(0, 2*math.pi)
    #alpha = 1
    #radius = rad_landing_1+350000 + plane_count*10000
    radius = random.uniform(rad_landing_1+350000, rad_landing_1+350000)
    #radius +=plane_count*10

    # Transform to cartesian and create position array
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    pos = np.array([x, y])

    # Generate plane
    size = sizes[plane_id]
    #size = random.randint(1,3)
    plane = Plane(pos, size, t, plane_id)
    print(plane.get_rho())
    print(size,'Vmin:',plane.Vmin(),'Vmax:',plane.Vmax(),'Vend:',plane.Vend(),'Vran:',plane.get_Vran())
    planes[plane_id] = plane
    plane_count += 1


# Get sequence minimizing time or fuel
def get_min(remaining, plane_id, sequence):
    sequence = copy.deepcopy(sequence)
    plane = planes[plane_id]
    if len(sequence['order']) == 0:

        sequence['total_time'] = max(plane.t_min(),separations[last_arrival_size][plane.size])
        V = plane.tar_dist() / sequence['total_time']
        pow =0.5 * plane.rho * V ** 3 * plane.S * plane.Cd0 + 2 * plane.W ** 2 / (plane.rho * V * plane.S * math.pi * plane.e * plane.A)
        sequence['total_fuel'] = pow * sequence['total_time']

    else:
        last = sequence['order'][-1]
        sequence['total_time'] = max(sequence['total_time']+separations[planes[last].size][plane.size], plane.t_min())
        V = plane.tar_dist() / sequence['total_time']
        pow = 0.5 * plane.rho * V ** 3 * plane.S * plane.Cd0 + 2 * plane.W ** 2 / (plane.rho * V * plane.S * math.pi * plane.e * plane.A)
        sequence['total_fuel'] += pow* sequence['total_time']

    # Update lists

    sequence['order'].append(plane_id)
    # Only way to copy a dictionary without the linked reference
    rem = copy.deepcopy(remaining)
    rem[plane.size].remove(plane_id)

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
        if (plane_id - rem[size][0]) < 30:

            get_min(rem, rem[size][0], sequence)


# Get all sequences (fuel, time, id, distance)
def get_sequence():

    # Declare grouped planes:  planes sorted for each size (1,2,3)
    grouped_planes = {1: {}, 2: {}, 3: {}}

    # Declare list of all ids
    all_seq = list(planes.keys())

    # Obtain first 12 aircraft (for DP only 12 aircraft can be sorted since the algorithm is O!) and increasing this number would increase exponentialy the computational time to establish the optimal sequences
    first_seq = all_seq[:12]
    last_seq = all_seq[12:]

    # Declare arrays of distance and size
    dists = []
    sizes = []

    # Define arrays of distance and size
    for plane in first_seq:
        sizes.append(planes[plane].size)
        dists.append(planes[plane].virtual_dist)

    # Set all planes with corresponding distance in grouped_planes
    for plane_id in first_seq:
        grouped_planes[planes[plane_id].size][plane_id] = planes[plane_id].virtual_dist

    # Sort each group of airplanes of the same size
    si1 = list(dict(sorted(grouped_planes[1].items(), key=operator.itemgetter(1))))
    si2 = list(dict(sorted(grouped_planes[2].items(), key=operator.itemgetter(1))))
    si3 = list(dict(sorted(grouped_planes[3].items(), key=operator.itemgetter(1))))

    # Define grouped_planes
    grouped_planes = {1: si1, 2: si2, 3: si3}
    #print(grouped_planes)
    # Initialize
    # if len(emergencies)>0:
    #     for emergency in emergencies:
    #         print('')
    #         #get_min(grouped_planes, emergency, {'order': [], 'total_time': 0, 'total_fuel': 0})
    # else:
    #     for size in grouped_planes:
    #         if len(grouped_planes[size]) != 0:
    #             print('')
    #             #get_min(grouped_planes, grouped_planes[size][0], {'order': [], 'total_time': 0, 'total_fuel': 0})
    #
    # seq_fuels = sorted([x['total_fuel'] for x in results])
    # seq_times = sorted([x['total_time'] for x in results])
    # for seq in results:
    #     if seq_fuels[0] == seq['total_fuel']:
    #
    #         sequences['fuel'] = list(seq['order'])+last_seq
    #
    #     if seq_times[0] == seq['total_time']:
    #         sequences['time'] = list(seq['order'])+last_seq
    # print('getmin', sequences['fuel'], get_fuel(sequences['fuel']))
    # sizes_out = []
    # for plane in sequences['fuel'][:10]:
    #     sizes_out.append(planes[plane].size)
    # with open('C:\\Users\\maxma\\PycharmProjects\\tf_test\\train_sequence.csv', 'a') as f:
    #     writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #
    #     #writer.whi(";".join(sizes)+';'+";".join(dists)+';'+";".join(sizes_out)+'\n')
    #     #writer = csv.writer(f)
    #     writer.writerow(sizes + dists + sizes_out)

    # Order by id ID

    # Declare id sequence (just by sorting keys)
    sequences['id'] =list(planes.keys())

    # Order by distance to target
    distances = {}

    # Determine distance to land of all planes
    for plane_id in planes:
        distances[plane_id] = planes[plane_id].tar_dist()

    # Sort planes by distances
    distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Initialize distance disquence
    sequences['distance'] = []

    # Declare distance sequence (by iterating over distances)
    for dist in distances:
        sequences['distance'].append(dist[0])
    global sequence

    sequence = copy.deepcopy(sequences['distance'])
    #print('dost',sequence)
    fuels = []
    sequences_fuel = []
    allplanes = sorted(sequence)
    print(sequence, get_fuel(sequence, True))
    sequence_pos = {}
    for plane in allplanes:
        sequence_pos[plane] = sequence.index(plane)
    sequence_pos = list(sequence_pos.values())
    sequence_pos_ga = list(sequence_pos)
    # print('seqpos', sequence_pos)
    # sequence_pos = optimize.fmin(get_fuel,sequence_pos)
    # print(sequence_pos)
    # sequence_pos = optimize.fmin(get_fuel, sequence_pos)
    # sequence_pos = optimize.fmin(get_fuel, sequence_pos)
    #
    # print('Sequence optimization:',get_fuel(sequence_pos),sequence_pos)



    # Genetic algorithm approach
    species = Species(400, 0.01, len(sequence_pos_ga))
    for i in range(400):
        species.next_generation()
        print(species.get_status())




    # Use the swap approach 10 times and get the best result
    for i in range(1, min(len(sequence),10)):
        # Start swapinng from position i (see DP_swap for detailed explanation)
        DP_swap_fuel(i, {}, i, sequence)

        # Define fuel consumed of sequence obtained
        fuels.append(get_fuel(sequence, True))
        #print('fuel:', get_time(sequence))

        # Append sequence
        sequences_fuel.append(sequence)

   # print(min(fuels))

    # Get sequence of minimum time consumed
    sequence = sequences_fuel[fuels.index(min(fuels))]
    print('Sequence Swap:',get_fuel(sequence, True),sequence)

    print(get_fuel(sequence, True))
    sequences['fuel'] = sequence


    sequence = copy.deepcopy(sequences['distance'])

    times = []
    sequences_time = []

    # Use the swap approach 10 times and get the best result
    for i in range(1, min(len(sequence),10)):
        # Start swapinng from position i (see DP_swap for detailed explanation)
        DP_swap_time(i, {}, i, sequence)

        # Define fuel consumed of sequence obtained
        times.append(get_time(sequence))

        # Append sequence
        sequences_time.append(sequence)


    # Get sequence of minimum fuel consumed
    sequence = sequences_time[times.index(min(times))]
    sequences['time'] = sequence

    set_velocities()


# Define velocities required for aircraft to land in the given sequence
def set_velocities():

    # Set count to one and establish virtual fuel count
    count = 0
    virt_fuel = 0

    # Iterate over sequence
    for plane_id in sequences[mode]:

        # If count is 0 there is no previous plane. (leading plane)
        if count == 0:

            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[plane_id].t_min(), separations[last_arrival_size][planes[plane_id].size])

            # Set required velocity to land in time
            planes[plane_id].vel = planes[plane_id].tar_dist() / planes[plane_id].t_exp

        else:
            # Time to land can be constrained by velocity (t_min) or by the separation between planes)
            planes[plane_id].t_exp = max(planes[previous_id].t_exp + separations[planes[previous_id].size][planes[plane_id].size],planes[plane_id].t_min())

            # Set required velocity to land in time
            planes[plane_id].vel = planes[plane_id].tar_dist() / planes[plane_id].t_exp

        # Set previous to current for the following iteration
        previous_id = plane_id
        count += 1
        virt_fuel +=  planes[plane_id].t_exp* planes[plane_id].get_pow()
    print(virt_fuel,'sequence',sequences[mode])

    for plane in planes.values():
        if np.linalg.norm(plane.pos) > 2*rad_turn +10:

            plane.get_path()

    for plane_id1 in planes:
        for plane_id2 in planes:
            if plane_id1 != plane_id2:
                xs1 = planes[plane_id1].xs
                xs2 = planes[plane_id2].xs
                min_l = min(len(xs1),len(xs2))
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

                    pygame.draw.circle(screen, yellow, cartesian_to_screen(point1), 5)
    pygame.display.flip()


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

# Airport dimensions
rad_landing_1 = 1000
rad_landing_2 = 200

# Performance NOT CHANGE for pre defined aircrafts, since t_exp depends on vmax
vel_min = 0.5
vel_max = 1

# Separation times for aircraft vortexs (first level leading, second level trailing)
separations = {
    1: {1: 82, 2: 69, 3: 60},
    2: {1: 131, 2: 69, 3: 60},
    3: {1: 196, 2: 157, 3: 66}
}

# Parameter to minimize
mode = 'time'

# Set clock to 0
t = 0
t_last = 0

# Declare list of emergency landings
emergencies = []

# Polygons
polygons = []

# Performance
rad_turn =40000

#
insert_delay(0,0,0)

# Planes
planes = {}
plane_count = 0

for i in range(0,pl_ini):
    generate_plane(plane_count)

for plane in planes:
    print(plane, planes[plane].size)
last_arrival_size = 1
sequence = list(np.arange(0,pl_ini))


landing = np.array([width/2, height/2])

pygame.init()

fps = 1400

fpsClock = pygame.time.Clock()


show_path = False
drawing = False
drawing_pol = []

pygame.font.init() # you have to call this at the start,
myfont = pygame.font.SysFont('Comic Sans MS', 15)
pygame.display.flip()

results = []
sequences= {}
fuel_total = 0
get_sequence()
dt =5
N = np.zeros((50, 50))

# Game loop
while len(planes) > 0:

    # Time step
    t += dt

    # Fill black screen
    screen.fill((0, 0, 0))

    # Draw elements
    button_heatmap =  pygame.Rect(200, 200, 50, 50)
    button_draw =  pygame.Rect(50, 50, 50, 50)
    pygame.draw.rect(screen, [255, 0, 0], button_heatmap)
    pygame.draw.rect(screen, [255, 0, 0], button_draw)

    # Set texts in screen
    textsurface = myfont.render('Plane count:'+str(plane_count), False, (100, 100, 100))
    screen.blit(textsurface, (0, 0))
    text_heat = myfont.render('Heat', False, (100, 100, 100))
    screen.blit(text_heat, (205, 225-7.5))
    text_draw = myfont.render('Draw', False, (100, 100, 100))
    screen.blit(text_draw, (55, 75 - 7.5))

    # Draw polygons
    for polygon in polygons:
        for p in range(0,len(polygon.points)):
            pygame.draw.line(screen, (255, 255, 255), cartesian_to_screen(polygon.points_dic[p]),cartesian_to_screen(polygon.points_dic[polygon.counterclockwise[p]]), 3)

    if len(drawing_pol)>1:
        for p in range(0,len(drawing_pol)-1):
            pygame.draw.line(screen, green, cartesian_to_screen(drawing_pol[p]),cartesian_to_screen(drawing_pol[p+1]), 3)

    # Detect events in game
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        # When click event
        if event.type == pygame.MOUSEBUTTONDOWN:

            # Record mouse position
            mouse_pos = event.pos

            # If heatmap clicked
            if button_heatmap.collidepoint(mouse_pos):

                # Draw heatmap (for collision visualization)
                power = 1
                rad = 100000
                lim = rad ** (1 / power)
                A = np.zeros((200, 200))
                for plane in planes:
                    for y in range(0, 800000, 4000):
                        for x in range(0, 800000, 4000):
                            if np.linalg.norm(np.array([x - 400000, y - 400000]) - planes[plane].pos) < rad:
                                A[int(y / 4000), int(x / 4000)] += -(lim * (np.linalg.norm(
                                    np.array([x - 400000, y - 400000]) - planes[plane].pos) / rad)) ** power + rad
                A = A[::-1, :]
                plt.imshow(A, interpolation='none')
                plt.show()

            # If draw button clicked
            if button_draw.collidepoint(mouse_pos):

                # If drawing enabled, draw new polygon and disable drawing
                if drawing:
                    polygons.append(Polygon(drawing_pol))
                    drawing_pol = []
                    for plane in planes.values():
                        plane.get_path()

                drawing = not drawing
            else:

                # If drawing enabled but mouse outside button
                if drawing:

                    # Append new point in polygon
                    drawing_pol.append(np.array(list(screen_to_cartesian(mouse_pos)*1000)))

    # Draw center screen
    pygame.draw.circle(screen, red, cartesian_to_screen([0,0]), 5)


    # For each plane
    for plane in planes.values():

        # Fuel consumed += power (proportional)
        fuel_total += plane.get_pow()*dt

        # Get new position of plane
        pos = plane.get_pos()

        # Draw plane
        pygame.draw.circle(screen, (244,244,244), cartesian_to_screen(pos), plane.size * 2)

        # Distance to land -= plane velocity
        plane.virtual_dist -= plane.vel*dt


        if int(t)%100 == 0:
            for y in range(0, 800000, 16000):
                for x in range(0, 800000, 16000):
                    break
                    #N[int(y / 16000), int(x / 16000)] += 100000000 / (np.linalg.norm(np.array([x - 400000 - pos[0], y - 400000 - pos[1], 11000])))

    if int(t) % 1000 == 0:
        print(graph.distances, 'yaaaaaaaaay')
        N_pic = N[::-1, :]

        plt.imshow(N_pic, interpolation='none')
        #plt.show()

    # Update distances and react for planes that arrived to target
    plane_ids = list(planes.keys())

    # For each plane
    for plane_id in plane_ids:
        # Update distance to target

        # In case airplane arrived to target
        if planes[plane_id].virtual_dist -planes[plane_id].vel < 0:

            # If plane is in emergency list, remove from emergencies
            if plane_id in emergencies:
                emergencies.remove(plane_id)

            # Print time since last landing, expected time, mode optimization, total fuel, total time, id and cost
            print('---------')
            print('Real:',t-t_last,'     Expected:', separations[last_arrival_size][planes[plane_id].size], '       Mode:',mode,'       Fuel:', fuel_total, '       Time:',t, '     ID:',plane_id, '        Cost:', fuel_total*20/1000000000)

            t_last = t

            # Define size of last landing aircraft
            last_arrival_size = planes[plane_id].size

            # Delete landing aircraft from list, generate new plane and delete aircraft from all frequencies
            del(planes[plane_id])
            del(sequence)
            if plane_count < len(sizes):
                generate_plane(plane_count)
            for seq_id in sequences:
                sequences[seq_id].remove(plane_id)
            results = []
            sequences = {}

            # Get new sequences
            get_sequence()

    # Draw screen
    pygame.display.flip()
    fpsClock.tick(fps)

# Close simulation
pygame.quit()
