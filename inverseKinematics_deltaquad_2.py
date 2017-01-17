import numpy as np
from visual import *
from myAnfis import anfis
import membership #import membershipfunction, mfDerivs
import timeit
import cPickle as pickle

def getAnglePlatform(x_pos, y_pos):
    with open('fuzzycontrol_normal_3.pkl', 'rb') as f:
        anf = pickle.load(f)

    input_val = np.array([[x_pos, y_pos]])

    anfis_matrix = anfis.predict(anf, input_val)
    return np.array([anfis_matrix[0,0], anfis_matrix[0, 1], anfis_matrix[0, 2]])

def getPlatformPositions(platform_normal, end_point):
    platform_radius = 25

    normal_vector = curve(pos=[(0,0,0), (10 *platform_normal)])

    z_axis = curve(pos=[(0, 0, 0), (0,0,-20)])

    beta = 180 - np.degrees(np.arctan2(platform_normal[0], platform_normal[2]))
    alpha = 180 - (np.degrees(np.arctan2(platform_normal[1], platform_normal[2])))
    gamma = 80 - (np.degrees(np.arctan2(platform_normal[0], platform_normal[1])))

    beta = -(90 - np.degrees(np.arctan2(np.sqrt(platform_normal[1] ** 2 + platform_normal[2] ** 2), platform_normal[0])))
    alpha = 90 - np.degrees(np.arctan2(np.sqrt(platform_normal[2] ** 2 + platform_normal[0] ** 2), platform_normal[1]))
    gamma = -(180 - np.degrees(np.arctan2(np.sqrt(platform_normal[0] ** 2 + platform_normal[1] ** 2), platform_normal[2])))

    print "Theta is:", beta, alpha, gamma

    rotate_x = np.matrix([[1, 0, 0, 0], [0, np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0],[0, np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0], [0, 0, 0, 1]])
    rotate_y = np.matrix([[np.cos(np.radians(beta)), 0, np.sin(np.radians(beta)), 0], [0, 1, 0, 0],[-np.sin(np.radians(beta)), 0, np.cos(np.radians(beta)), 0], [0, 0, 0, 1]])
    rotate_z = np.matrix([[np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0, 0],[np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    translate = np.matrix([[1, 0, 0, end_point[0]], [0, 1, 0, end_point[1]], [0, 0, 1, end_point[2]], [0, 0, 0, 1]])

    # define the position where the arms meet the base
    platform_1 = np.array([platform_radius * np.cos(np.deg2rad(90)), platform_radius * np.sin(np.deg2rad(90)), 0])
    platform_2 = np.array([platform_radius * np.cos(np.deg2rad(210)), platform_radius * np.sin(np.deg2rad(210)), 0])
    platform_3 = np.array([platform_radius * np.cos(np.deg2rad(330)), platform_radius * np.sin(np.deg2rad(330)), 0])

    # draw lines to platform
    # arm_link_1 = curve(pos=[(platform_1), (platform_2)])
    # arm_link_2 = curve(pos=[(platform_2), (platform_3)])
    # arm_link_3 = curve(pos=[(platform_3), (platform_1)])

    p1 = np.matrix([[platform_1[0]], [platform_1[1]], [platform_1[2]], [1]])
    p2 = np.matrix([[platform_2[0]], [platform_2[1]], [platform_2[2]], [1]])
    p3 = np.matrix([[platform_3[0]], [platform_3[1]], [platform_3[2]], [1]])

    new_p1 = translate * rotate_z * rotate_y * rotate_x * p1
    new_p2 = translate * rotate_z * rotate_y * rotate_x * p2
    new_p3 = translate * rotate_z * rotate_y * rotate_x * p3

    new_p1_array = np.array([new_p1[0, 0], new_p1[1, 0], new_p1[2, 0]])
    new_p2_array = np.array([new_p2[0, 0], new_p2[1, 0], new_p2[2, 0]])
    new_p3_array = np.array([new_p3[0, 0], new_p3[1, 0], new_p3[2, 0]])

    # draw lines to platform
    arm_link_11 = curve(pos=[(new_p1_array), (new_p2_array)])
    arm_link_22 = curve(pos=[(new_p2_array), (new_p3_array)])
    arm_link_33 = curve(pos=[(new_p3_array), (new_p1_array)])

    return new_p1_array, new_p2_array, new_p3_array

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


pos = np.array([0, 0, -124])

normal_vector = getAnglePlatform(pos[0], pos[1])
print "The normal vector is", normal_vector

new_p1, new_p2, new_p3 = getPlatformPositions(normal_vector, pos)

arm = 110
leg = 266.7
# length from center of base to axis of rotation
base_radius = 55

# define the position where the arms meet the base
base_1 = np.array([base_radius * np.cos(np.deg2rad(90)), base_radius * np.sin(np.deg2rad(90)), 0])
base_2 = np.array([base_radius * np.cos(np.deg2rad(210)), base_radius * np.sin(np.deg2rad(210)), 0])
base_3 = np.array([base_radius * np.cos(np.deg2rad(330)), base_radius * np.sin(np.deg2rad(330)), 0])
# draw lines to platform
arm_link_1 = curve(pos=[(base_1), (base_2)])
arm_link_2 = curve(pos=[(base_2), (base_3)])
arm_link_3 = curve(pos=[(base_3), (base_1)])

# Sphere Circle Intersection
# http://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection

c_c = base_1
print "c_c", c_c
c_s = new_p1
print "c_s", c_s
n = np.array([1,0,0]) # normal vector to circle

d = np.dot(n, c_c - c_s)

if abs(d) > leg:
    print "No intersection"

if d == leg:
    print "Only one intersection"

c_p = c_s + (d * n)     # center of sphere circle cut by plane of intersection
print "c_p", c_p

r_p = np.sqrt(leg ** 2 - d ** 2)    # radius of sphere circle cut by plane

d_p = np.sqrt((c_c[0] - c_p[0])**2 + (c_c[1] - c_p[1])**2 + (c_c[2] - c_p[2])**2)

# solve for line of intersection between circles, find the center and radius
# h = 0.5 + (arm**2 - leg**2) / (d_p**2)
# a = (arm**2 - leg**2 + d_p**2) / (2 * d_p)
# h = np.sqrt(leg**2 - a**2)
# print "h", h

dy = c_p[1] - c_c[1]
dz = c_p[2] - c_c[2]
d = np.sqrt(dy*dy+dz*dz)

a = (arm*arm-leg*leg+d_p*d_p)/(2*d_p)
h = np.sqrt(arm*arm-a*a)
ym = c_c[1] + a*dy/d
zm = c_c[2] + a*dz/d
ys1 = ym + h*dz/d
ys2 = ym - h*dz/d
zs1 = zm - h*dy/d
zs2 = zm + h*dy/d

K1 = np.array([c_c[0], ys1, zs1])
print K1

# circle_conn_vector = unit_vector(c_p - c_c)
#
# c_i = c_c + (h * (circle_conn_vector))
# print "c_i", c_i
#
# r_i = np.sqrt(leg**2 - (h**2))
#
# t = np.cross(c_p-c_c, n)
# print "t", t
# t = unit_vector(t)
# print "t", t
#
# p_0 = c_i - (t * r_i)
# p_1 = c_i + (t * r_i)
# print "Points: ", p_0, p_1
#
# K1 = p_0

platform_k1 = curve(pos=[(K1), (new_p1)])
base_k1 = curve(pos=[(K1), (base_1)])

leg_dist = np.sqrt((K1[0]-new_p1[0])**2 + (K1[1]-new_p1[1])**2 + (K1[2]-new_p1[2])**2)
print leg_dist

arm_dist = np.sqrt((K1[0]-base_1[0])**2 + (K1[1]-base_1[1])**2 + (K1[2]-base_1[2])**2)
print arm_dist









# new_p1_v = np.array(new_p1 - pos)
# new_p2_v = np.array(new_p2 - pos)
# new_p3_v = np.array(new_p3 - pos)
# platform_v1 = curve(pos=[(pos), (new_p1)])
# platform_v2 = curve(pos=[(pos), (new_p2)])
# platform_v3 = curve(pos=[(pos), (new_p3)])
#
# new_p1_v = new_p1_v/np.linalg.norm(new_p1_v)
# new_p2_v = new_p2_v/np.linalg.norm(new_p2_v)
# new_p3_v = new_p3_v/np.linalg.norm(new_p3_v)
#
# new_p1_cross = np.cross(normal_vector, new_p1_v)
# new_p2_cross = np.cross(normal_vector, new_p2_v)
# new_p3_cross = np.cross(normal_vector, new_p3_v)
#
# line_dir = np.dot(new_p1, new_p1_cross)
# print "line dir", line_dir
#
#
# a = (line_dir / new_p1_cross[1])
# b = -(new_p1_cross[2] / new_p1_cross[1])
#
# A = (b ** 2) + 1
# B = (2 * a * b) - (2 * b * new_p1[1]) - (2 * new_p1[2])
# C = (new_p1[0] ** 2) + (a ** 2) - (2 * a * new_p1[1]) + (new_p1[1] ** 2) + (new_p1[2] ** 2) - (leg ** 2)
#
# print a, b
# print A, B, C
#
# z = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
# print "z", z
#
# y = a + (b * z)
# print "y", y
#
# K1 = np.array([0, y, z])
#
# platform_k1 = curve(pos=[(K1), (new_p1)])
# base_k1 = curve(pos=[(K1), (base_1)])
#
# leg_dist = np.sqrt((K1[0]-new_p1[0])**2 + (K1[1]-new_p1[1])**2 + (K1[2]-new_p1[2])**2)
# print leg_dist
#
# arm_dist = np.sqrt((K1[0]-base_1[0])**2 + (K1[1]-base_1[1])**2 + (K1[2]-base_1[2])**2)
# print arm_dist

# arm = 100
# leg = 180
# foot_height = 20
#
# x0 = 0
# y0 = 0
# z0 = 0
#
# P_x = 20
# P_y = -20
# P_z = -130
#
# dist_platform_foot = np.sqrt((P_x - x0)**2 + (P_y - y0)**2 + (P_z - z0)**2)
#
# # main_vector = np.array([P_x-x0, P_y - y0, P_z - z0])
# # main_vector = np.array([x0-P_x, y0-P_y, z0-P_z])
# # print "main_vector", main_vector
#
#
# # length from center of base to axis of rotation
# base_radius = 55
#
# # define the position where the arms meet the base
# base_1 = np.array([base_radius * np.cos(np.deg2rad(90)), base_radius * np.sin(np.deg2rad(90)), 0])
# base_2 = np.array([base_radius * np.cos(np.deg2rad(210)), base_radius * np.sin(np.deg2rad(210)), 0])
# base_3 = np.array([base_radius * np.cos(np.deg2rad(330)), base_radius * np.sin(np.deg2rad(330)), 0])
#
# # draw lines to platform
# arm_link_1 = curve(pos=[(base_1), (base_2)])
# arm_link_2 = curve(pos=[(base_2), (base_3)])
# arm_link_3 = curve(pos=[(base_3), (base_1)])
#
#
# # length from center of platform (foot) to axis of rotation
# foot_radius = 25
# # define the position where the arms meet the base
# platform_1 = np.array([foot_radius * np.cos(np.deg2rad(90)), foot_radius * np.sin(np.deg2rad(90)), -dist_platform_foot + foot_height])
# platform_2 = np.array([foot_radius * np.cos(np.deg2rad(210)), foot_radius * np.sin(np.deg2rad(210)), -dist_platform_foot + foot_height])
# platform_3 = np.array([foot_radius * np.cos(np.deg2rad(330)), foot_radius * np.sin(np.deg2rad(330)), -dist_platform_foot + foot_height])
#
# # draw lines to show platform
# # arm_link_1 = curve(pos=[(platform_1), (platform_2)])
# # arm_link_2 = curve(pos=[(platform_2), (platform_3)])
# # arm_link_3 = curve(pos=[(platform_3), (platform_1)])
#
# print "base_p1", base_1
# p = np.matrix([[0], [0], [-dist_platform_foot + foot_height], [1]])
# p1 = np.matrix([[platform_1[0]], [platform_1[1]], [platform_1[2]], [1]])
# p2 = np.matrix([[platform_2[0]], [platform_2[1]], [platform_2[2]], [1]])
# p3 = np.matrix([[platform_3[0]], [platform_3[1]], [platform_3[2]], [1]])
#
# alpha = 90 + np.degrees(np.arctan2(P_z, P_y))
# beta = 90 + np.degrees(np.arctan2(P_z, P_x))
# gamma = 0
#
# rotate_x = np.matrix([[1, 0, 0, 0], [0, np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0], [0, np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0], [0, 0, 0, 1]])
# rotate_y = np.matrix([[np.cos(np.radians(beta)), 0, np.sin(np.radians(beta)), 0], [0, 1, 0 , 0], [-np.sin(np.radians(beta)), 0, np.cos(np.radians(beta)), 0], [0, 0, 0, 1]])
# rotate_z = np.matrix([[np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0, 0], [np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#
# new_p = rotate_y * rotate_x * p
# new_p1 = rotate_y * rotate_x * p1
# new_p2 = rotate_y * rotate_x * p2
# new_p3 = rotate_y * rotate_x * p3
#
# print "new_p1", new_p1
# new_p = np.array([new_p[0, 0], new_p[1, 0], new_p[2, 0]])
# main_vector_line = curve(pos=[(new_p), ([0, 0, 0])])
#
# new_p1 = np.array([new_p1[0, 0], new_p1[1, 0], new_p1[2, 0]])
# new_p2 = np.array([new_p2[0, 0], new_p2[1, 0], new_p2[2, 0]])
# new_p3 = np.array([new_p3[0, 0], new_p3[1, 0], new_p3[2, 0]])
#
# print "newp1", new_p1
#
# # draw lines to show arms
# arm_link_4 = curve(pos=[(new_p1), (new_p2)])
# arm_link_5 = curve(pos=[(new_p2), (new_p3)])
# arm_link_6 = curve(pos=[(new_p3), (new_p1)])
#
# new_p1_v = np.array(new_p1 - new_p)
# print "new_p1_v", new_p1_v
# new_p2_v = np.array(new_p2 - new_p)
# new_p3_v = np.array(new_p3 - new_p)
# platform_v1 = curve(pos=[(new_p), (new_p1)])
# platform_v2 = curve(pos=[(new_p), (new_p2)])
# platform_v3 = curve(pos=[(new_p), (new_p3)])
#
# main_vector = np.array([x0-new_p[0], y0-new_p[1], z0-new_p[2]])
# print "main_vector", main_vector
#
# main_vector = main_vector/np.linalg.norm(main_vector)
# # new_p1_v = new_p1_v/np.linalg.norm(new_p1_v)
# new_p2_v = new_p2_v/np.linalg.norm(new_p2_v)
# new_p3_v = new_p3_v/np.linalg.norm(new_p3_v)
#
# new_p1_cross = np.cross(main_vector, new_p1_v)
# new_p2_cross = np.cross(main_vector, new_p2_v)
# new_p3_cross = np.cross(main_vector, new_p3_v)
#
# print "new_p1_cross", new_p1_cross
#
# platform_cross1 = curve(pos=[(new_p1 + new_p1_cross), (new_p1)])
# platform_cross2 = curve(pos=[(new_p2 + new_p2_cross), (new_p2)])
# platform_cross3 = curve(pos=[(new_p3 + new_p3_cross), (new_p3)])
#
#
# line_dir = np.dot(new_p1, new_p1_cross)
# print "line dir", line_dir
#
#
