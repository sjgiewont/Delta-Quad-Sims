import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm
import csv
# import sqlite3
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# Find the intersection of three spheres
# P1,P2,P3 are the centers, r1,r2,r3 are the radii
def trilaterate(P1,P2,P3,r1,r2,r3):
    temp1 = P2-P1
    e_x = temp1/norm(temp1)
    temp2 = P3-P1
    i = dot(e_x,temp2)
    temp3 = temp2 - i*e_x
    e_y = temp3/norm(temp3)
    e_z = cross(e_x,e_y)
    d = norm(P2-P1)
    j = dot(e_y,temp2)
    x = (r1*r1 - r2*r2 + d*d) / (2*d)
    y = (r1*r1 - r3*r3 -2*i*x + i*i + j*j) / (2*j)
    temp4 = r1*r1 - x*x - y*y
    if temp4<0:
        raise Exception("The three spheres do not intersect!");
    z = sqrt(temp4)
    p_12_a = P1 + x*e_x + y*e_y + z*e_z
    p_12_b = P1 + x*e_x + y*e_y - z*e_z
    return p_12_a,p_12_b

def forwardKinematics(theta_1, theta_2, theta_3):
    # length from center of base to axis of rotation
    base_radius = 55
    # length from center of platform (foot) to axis of rotation
    foot_radius = 25
    # length of arm and leg
    arm = 110
    leg = 266.7

    # define the position where the arms meet the base
    base_p1 = np.array([base_radius * np.cos(np.deg2rad(90)), base_radius * np.sin(np.deg2rad(90)), 0])
    base_p2 = np.array([base_radius * np.cos(np.deg2rad(210)), base_radius * np.sin(np.deg2rad(210)), 0])
    base_p3 = np.array([base_radius * np.cos(np.deg2rad(330)), base_radius * np.sin(np.deg2rad(330)), 0])

    # the first knee, in the universal y,z plane, pointing towards 12 o'clock
    k1 = base_p1 + np.array([0, -arm * np.cos(np.deg2rad(theta_1)), arm * np.sin(np.deg2rad(theta_1))])

    # second knee, pointing near 7 o'clock
    k2 = base_p2 + np.array([0, -arm * np.cos(np.deg2rad(theta_2)), arm * np.sin(np.deg2rad(theta_2))])
    k2 = np.array([k2[0] * np.cos(np.deg2rad(120)) - k2[1] * np.sin(np.deg2rad(120)) + base_p2[0] - np.cos(np.deg2rad(120)) * base_p2[0] + np.sin(np.deg2rad(120)) * base_p2[1], k2[0] * np.sin(np.deg2rad(120)) + k2[1] * np.cos(np.deg2rad(120)) + base_p2[1] - np.sin(np.deg2rad(120)) * base_p2[0] - np.cos(np.deg2rad(120)) * base_p2[1], k2[2]])

    # third knee, pointing near 5 o'clock
    k3 = base_p3 + np.array([0, -arm * np.cos(np.deg2rad(theta_3)), arm * np.sin(np.deg2rad(theta_3))])
    k3 = np.array([k3[0] * np.cos(np.deg2rad(240)) - k3[1] * np.sin(np.deg2rad(240)) + base_p3[0] - np.cos(np.deg2rad(240)) * base_p3[0] + np.sin(np.deg2rad(240)) * base_p3[1], k3[0] * np.sin(np.deg2rad(240)) + k3[1] * np.cos(np.deg2rad(240)) + base_p3[1] - np.sin(np.deg2rad(240)) * base_p3[0] - np.cos(np.deg2rad(240)) * base_p3[1], k3[2]])

    # distances between knees
    dist_k1_k2 = np.linalg.norm(k1 - k2)
    dist_k1_k3 = np.linalg.norm(k1 - k3)
    dist_k2_k3 = np.linalg.norm(k2 - k3)

    # find the intersection of the three arm spheres
    circle_intersect_1, circle_intersect_2 = trilaterate(k1, k2, k3, leg, leg, leg)

    # find the distance between intersection point and the platform
    height_knee_2_intersect_1 = np.sqrt(leg * leg - 0.25 * (dist_k1_k2 * dist_k1_k2))
    height_knee_2_intersect_2 = np.sqrt(leg * leg - 0.25 * (dist_k1_k3 * dist_k1_k3))
    height_knee_2_intersect_3 = np.sqrt(leg * leg - 0.25 * (dist_k2_k3 * dist_k2_k3))
    height_knee_2_intersect = np.mean([height_knee_2_intersect_1, height_knee_2_intersect_2, height_knee_2_intersect_3])

    height_knee_2_platform_1 = np.sqrt(leg * leg - 0.25 * (dist_k1_k2 - foot_radius) * (dist_k1_k2 - foot_radius))
    height_knee_2_platform_2 = np.sqrt(leg * leg - 0.25 * (dist_k1_k3 - foot_radius) * (dist_k1_k3 - foot_radius))
    height_knee_2_platform_3 = np.sqrt(leg * leg - 0.25 * (dist_k2_k3 - foot_radius) * (dist_k2_k3 - foot_radius))
    height_knee_2_platform = np.mean([height_knee_2_platform_1, height_knee_2_platform_2, height_knee_2_platform_3])

    height_intersect_2_platform = height_knee_2_platform - height_knee_2_intersect

    # Using vectors, find point along a line, given a distance.
    vx_circle_intersect = circle_intersect_2[0] - circle_intersect_1[0]
    vy_circle_intersect = circle_intersect_2[1] - circle_intersect_1[1]
    vz_circle_intersect = circle_intersect_2[2] - circle_intersect_1[2]

    vmag = np.sqrt(vx_circle_intersect * vx_circle_intersect + vy_circle_intersect * vy_circle_intersect + vz_circle_intersect * vz_circle_intersect)

    vx = vx_circle_intersect / vmag
    vy = vy_circle_intersect / vmag
    vz = vz_circle_intersect / vmag

    # foot_pt_x = round((circle_intersect_2[0] + vx * (height_intersect_2_platform)), 1)
    # foot_pt_y = round((circle_intersect_2[1] + vy * (height_intersect_2_platform)), 1)
    # foot_pt_z = round((circle_intersect_2[2] + vz * (height_intersect_2_platform)), 1)

    foot_pt_x = circle_intersect_2[0] + vx * (height_intersect_2_platform)
    foot_pt_y = circle_intersect_2[1] + vy * (height_intersect_2_platform)
    foot_pt_z = circle_intersect_2[2] + vz * (height_intersect_2_platform)

    # foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z, theta_1, theta_2, theta_3])
    # foot_pt = [foot_pt_x, foot_pt_y, foot_pt_z, theta_1, theta_2, theta_3]

    foot_pt = [foot_pt_x, foot_pt_y, foot_pt_z, theta_1, theta_2, theta_3, vx, vy, vz]

    return foot_pt


theta_low = 140
theta_high = 222
step = 2

with open('table_140_220_two.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # headers = ['x', 'y', 'z', 'theta1', 'theta2', 'theta3']
    # spamwriter.writerow(headers)
    for theta1 in np.arange(theta_low, theta_high, step):
        print theta1
        for theta2 in np.arange(theta_low, theta_high, step):
            for theta3 in np.arange(theta_low, theta_high, step):
                foot_pos = forwardKinematics(theta1, theta2, theta3)
                # if foot_pos[2] == -124.4:
                # if foot_pos[2] <= -175 and foot_pos[2] >= -170:
                spamwriter.writerow(foot_pos)

# theta_low = 180
# theta_high = 220
# step = 1
#
# X =[]
# Y = []
# Z = []
#
# for theta1 in np.arange(theta_low, theta_high, step):
#     for theta2 in np.arange(theta_low, theta_high, step):
#         for theta3 in np.arange(theta_low, theta_high, step):
#             foot_pos = forwardKinematics(theta1, theta2, theta3)
#             X.append(theta1)
#             Y.append(theta2)
#             Z.append(foot_pos[1])
#
# print "drawing"
# # X, Y = np.meshgrid(X, Y)
# print X
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(X, Y, Z)
# plt.show()
