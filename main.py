from inverseKinematics import *
import Queue
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit


def main():
    global leg1_q, leg2_q
    ideal_arm = []
    ideal_leg = []

    arm = 10
    leg = 20

    first_pos = [0, 0, 0]
    new_pos = [2, 2, 0]

    first_pos2 = [1, 1, 0]
    new_pos2 = [3, 3, 0]

    #step(curr_pos, new_pos)

    leg1_q = Queue.Queue(maxsize=0)
    leg2_q = Queue.Queue(maxsize=0)

    curr_pos = step_2(leg1_q, first_pos, new_pos, 5)
    curr_pos = step_2(leg2_q, first_pos2, new_pos2, 5)

    while (leg1_q.empty() != True or leg2_q.empty() != True):
        print leg1_q.get()
        print leg2_q.get()


    #curr_pos = step_2(curr_pos, first_pos, 0)


def step_2(leg, curr_pos, new_pos, step_height):
    global leg1_q
    #parabola function between 2 points
    start = timeit.default_timer()

    start_pt = np.array([curr_pos[0], curr_pos[1], curr_pos[2]])
    end_pt = np.array([new_pos[0], new_pos[1], new_pos[2]])

    #step_height = 5     #the height difference between steps, relative offset from start Z
    t = np.linspace(0, 1, 10)

    mid = (start_pt + end_pt) / float(2)
    #mid = (curr_pos + new_pos) / float(2)

    mid_pt = np.array([mid[0], mid[1], mid[2]+step_height])

    x_pts = np.matrix([[curr_pos[0]], [mid_pt[0]], [new_pos[0]]])
    y_pts = np.matrix([[curr_pos[1]], [mid_pt[1]], [new_pos[1]]])
    z_pts = np.matrix([[curr_pos[2]], [mid_pt[2]], [new_pos[2]]])

    A_1 = np.matrix([[2, -4, 2], [-3, 4, -1], [1, 0, 0]])

    x_coeff = A_1 * x_pts
    y_coeff = A_1 * y_pts
    z_coeff = A_1 * z_pts

    x = x_coeff.item(0)*t*t + x_coeff.item(1)*t + x_coeff.item(2)
    y = y_coeff.item(0)*t*t + y_coeff.item(1)*t + y_coeff.item(2)
    z = z_coeff.item(0)*t*t + z_coeff.item(1)*t + z_coeff.item(2)

    stop = timeit.default_timer()
    print "The Time:", stop - start

    # add the numpy array to the queue
    map(leg.put, x)

    return new_pos


if __name__ == "__main__":
    main()

