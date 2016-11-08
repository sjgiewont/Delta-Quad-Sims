from inverseKinematics import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import timeit

ideal_arm = []
ideal_leg = []

arm = 10
leg = 20

# start = timeit.default_timer()
#
# for num in range(0,100):
#     for i in range(10,22):
#         theta1, theta2, theta3 = inverseKinematic(0, 0, i, arm, leg)
#
# stop = timeit.default_timer()
# print stop - start

#parabola function between 2 points

start_pt = np.array([0, 0, 0])
end_pt = np.array([2, 5, 0])

step_height = 3     #the height difference between steps, relative offset from start Z

# mid_x = (start_pt[0] + end_pt[0]) / float(2)
# mid_y = (start_pt[1] + end_pt[1]) / float(2)
mid = (start_pt + end_pt) / float(2)

mid_pt = np.array([mid[0], mid[1], mid[2]+step_height])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# for t in np.arange(0, 1, 0.001):
t = np.arange(0, 1, 0.001)

start = timeit.default_timer()

x_pt = start_pt[0] - (t * (3*start_pt[0] - 4*mid_pt[0] + end_pt[0])) + 2*t*t*(start_pt[0] - 2*mid_pt[0] + end_pt[0])
y_pt = start_pt[1] - (t * (3*start_pt[1] - 4*mid_pt[1] + end_pt[1])) + 2*t*t*(start_pt[1] - 2*mid_pt[1] + end_pt[1])
z_pt = start_pt[2] - (t * (3*start_pt[2] - 4*mid_pt[2] + end_pt[2])) + 2*t*t*(start_pt[2] - 2*mid_pt[2] + end_pt[2])

stop = timeit.default_timer()
print stop - start

#ax.scatter(sample_pt[0], sample_pt[1], sample_pt[2])
ax.scatter(x_pt, y_pt, z_pt)



plt.show()




# for t in np.arange(0.01, 1, 0.001):
#     # Pc = (start_pt * t * t) + (mid_pt * 2 * t * (1 - t)) + (end_pt * (1 - t) * (1 - t))
#     # curr_pos = (Pc - (start_pt * t * t) - (end_pt * t * t)) / t
#
#     curr_pos = (mid_pt / (2 * t * (1 - t))) - ((start_pt * t) / (2 * (1 -t))) - ((end_pt * (1 - t)) / (2 * t))
#
#     print curr_pos

#
# for t in np.arange(0.01, 1, 0.001):
#     t_1 = t
#     a2 = ((mid_pt - start_pt) - (t_1 * (end_pt - start_pt))) / (t_1 * (t_1 - 1))
#     a1 = end_pt - start_pt - a2
#     curr_pos = (a2 * t_1) + (a1 * t_1) + start_pt
#     print curr_pos
#     plt.scatter(curr_pos[0], curr_pos[1])
#
# plt.show()


# done, x0, y0, z0 = forward(45, -45, -45, arm, leg)
# done, x1, y1, z1 = forward(45, 45, 45, arm, leg)
#
# print abs(y0*2)
# print z1
# for arm in range(5,17):
#     for leg in range(12, 52):
#         x, y, z = maxWorkspace(arm, leg)
#         ideal_arm.append(x)
#         ideal_arm.append(y)
#
# print map(ideal_arm)

# theta_array_1 = np.asarray(theta_list_1)
# theta_array_2 = np.asarray(theta_list_2)
# theta_array_3 = np.asarray(theta_list_3)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(theta_array_1, theta_array_2, theta_array_3)
# plt.show()
