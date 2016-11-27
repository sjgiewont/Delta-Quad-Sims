from inverseKinematics import *
from piecewiseMotion import *
import Queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit
import thread
import threading


def main():
    global leg1_q, leg2_q, leg3_q, leg4_q
    ideal_arm = []
    ideal_leg = []

    arm = 10
    leg = 20

    position_list = np.matrix([[2, 0, 0], [-2, 0, 0], [2, 0, 0]])

    # print pos_list.item(1)
    # print pos_list[1,:]

    first_pos = [0, 0, 0]
    new_pos = [2, 0, 0]

    first_pos2 = [0, 0, 0]
    new_pos2 = [-1, 0, 0]

    first_pos3 = [0, 0, 0]
    new_pos3 = [2, 0, 0]

    first_pos4 = [0, 1, 0]
    new_pos4 = [-1, 3, 0]

    # step(curr_pos, new_pos)

    leg1_q = Queue.Queue(maxsize=0)
    leg2_q = Queue.Queue(maxsize=0)
    leg3_q = Queue.Queue(maxsize=0)
    leg4_q = Queue.Queue(maxsize=0)

    curr_pos1 = step_to(leg1_q, first_pos, new_pos, 5)
    curr_pos2 = step_to(leg2_q, first_pos2, new_pos2, 5)
    curr_pos3 = step_to(leg3_q, first_pos3, new_pos3, 5)
    curr_pos4 = step_to(leg4_q, first_pos4, new_pos4, 5)

    #thread.start_new_thread(add_leg1(), None)
    leg_thread = threading.Thread(target=add_leg1, args=[first_pos, new_pos4])
    leg_thread.start()

    walk_dir(0, 2, 100)

    # thread to continually check for user input

    # need function for walking
    # def walk(direction, speed):
    #   global curr_pos
    #   step1 = [1, 0, 0]
    #   step2 = [-1, 0, 0]
    #
    #   step_to(leg, curr_pos, new_pos, step_height)
    #

    # execute walking trajectory
    while (leg1_q.empty() != True or leg2_q.empty() != True):
        leg1_pos = leg1_q.get(2)
        print leg1_pos[0]
        # print leg2_q.get()
        # print leg3_q.get()
        # print leg4_q.get()

# walking gait, can walk in a direction and particular number of steps.
# The precision is the number incremental steps between the start and stop motions
def walk_dir(degrees, step_num, precision):
    #calculate the walking trajectory of one step
    walking_trajectory = piecewiseMotion(degrees, precision)

    # initialize the index of each leg, offset all of them
    FL_leg_index = 0
    FR_leg_index = 25
    HL_leg_index = 50
    HR_leg_index = 75

    steps = 0

    # step a certain amount of times
    while(steps < step_num):
        print "Front Left:", walking_trajectory[FL_leg_index]
        print "Front Right:", walking_trajectory[FR_leg_index]
        print "Hind Left:", walking_trajectory[HL_leg_index]
        print "Hind Right:", walking_trajectory[HR_leg_index]

        FL_leg_index += 1
        FR_leg_index += 1
        HL_leg_index += 1
        HR_leg_index += 1

        # if reached index limit, loop back and restart index count
        if FL_leg_index >= precision:
            FL_leg_index = 0
            steps += 1                  # keep track of the number of steps taken

        if FR_leg_index >= precision:
            FR_leg_index = 0

        if HL_leg_index >= precision:
            HL_leg_index = 0

        if HR_leg_index >= precision:
            HR_leg_index = 0




# parabola function between 2 points
def step_to(leg, curr_pos, new_pos, step_height):
    global leg1_q

    # start a timer for benchmarking purposes
    start = timeit.default_timer()

    # convert python array to numpy array to streamline math
    start_pt = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
    end_pt = np.array([new_pos[0], new_pos[1], new_pos[2]])

    # generate numpy array of numbers 0 to 1, to be used in parametric equations
    t = np.linspace(0, 1, 20)

    # determine the mean of the start and end points
    mid = (start_pt + end_pt) / float(2)

    # determine the actual mid point with the step_height factored in
    mid_pt = np.array([mid[0], mid[1], mid[2] + step_height])

    # create numpy matrix of all x, y z points
    x_pts = np.matrix([[curr_pos[0]], [mid_pt[0]], [new_pos[0]]])
    y_pts = np.matrix([[curr_pos[1]], [mid_pt[1]], [new_pos[1]]])
    z_pts = np.matrix([[curr_pos[2]], [mid_pt[2]], [new_pos[2]]])

    # generate the standard inverse matrix to solve parabolic constraints
    A_1 = np.matrix([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])

    # solve all coefficients by multiplying inverse with points
    x_coeff = A_1 * x_pts
    y_coeff = A_1 * y_pts
    z_coeff = A_1 * z_pts

    # plug in solved coefficents to determine parametric equation for each axis
    x = x_coeff.item(0)*t*t + x_coeff.item(1)*t + x_coeff.item(2)
    y = y_coeff.item(0)*t*t + y_coeff.item(1)*t + y_coeff.item(2)
    z = z_coeff.item(0)*t*t + z_coeff.item(1)*t + z_coeff.item(2)

    pos = []

    # create matrix of all positions along trajectory
    for i in range(len(x)):
        pos.append([x[i], y[i], z[i]])

    # add each row of the matrix to the queue
    map(leg.put, pos)

    # stop the timer to for benchmarking purposes
    stop = timeit.default_timer()
    print "The Time:", stop - start

    # return the final position
    return new_pos


def add_leg1(first_pos, new_pos):
    #args first_pos and new_pos must be arrays

    curr_pos1 = step_to(leg1_q, first_pos, new_pos, 5)
    return

if __name__ == "__main__":
    main()