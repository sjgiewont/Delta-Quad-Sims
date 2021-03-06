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

    normal_vector = curve(pos=[(0,0,0), (10 * platform_normal)])

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

    # new_p1 = translate * rotate_z * rotate_y * rotate_x * p1
    # new_p2 = translate * rotate_z * rotate_y * rotate_x * p2
    # new_p3 = translate * rotate_z * rotate_y * rotate_x * p3

    new_p1 = translate * rotate_y * rotate_x * p1
    new_p2 = translate * rotate_y * rotate_x * p2
    new_p3 = translate * rotate_y * rotate_x * p3

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

def euclidean_distance(P1, P2):
    return np.sqrt((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2 + (P1[2] - P2[2]) ** 2)


def getKneePosition(base_p, new_p):
    # Sphere Circle Intersection
    # http://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection
    arm = 110
    leg = 266.7
    c_c = base_p
    print "c_c", c_c
    c_s = new_p
    print "c_s", c_s
    z_vector = np.array([0, 0, 1])

    n = np.cross(unit_vector(base_p), z_vector)
    print "n: ", n

    d = np.dot(c_c - c_s, n)
    print "d: ", d

    if abs(d) > leg:
        print "No intersection"

    if d == leg:
        print "Only one intersection"

    c_p = c_s + (d * n)  # center of sphere circle cut by plane of intersection
    print "c_p", c_p

    r_p = np.sqrt(leg ** 2 - d ** 2)  # radius of sphere circle cut by plane
    print "r_p: ", r_p

    d_p = np.sqrt((c_c[0] - c_p[0]) ** 2 + (c_c[1] - c_p[1]) ** 2 + (c_c[2] - c_p[2]) ** 2)
    print "d_p: ", d_p

    # solve for line of intersection between circles, find the center and radius
    # h = 0.5 + (arm**2 - r_p**2) / (d_p**2)
    # print "h", h

    h = (arm **2 - r_p**2 + d_p**2) / (2 * d_p)
    print "H: ", h

    c_i = base_p + (h * unit_vector(c_p - base_p))
    print "c_i: ", c_i
    r_i = np.sqrt(arm**2 - h**2)
    print "r_i: ", r_i

    t = unit_vector(np.cross(c_p - c_c, n))
    print "t: ", t

    p0 = c_i - (t * r_i)
    p1 = c_i + (t * r_i)

    # dy = c_p[1] - c_c[1]
    # dz = c_p[2] - c_c[2]
    # d = np.sqrt(dy * dy + dz * dz)
    #
    # a = (arm * arm - r_p * r_p + d_p * d_p) / (2 * d_p)
    # h = np.sqrt(arm * arm - a * a)
    # ym = c_c[1] + a * dy / d
    # zm = c_c[2] + a * dz / d
    # ys1 = ym + h * dz / d
    # ys2 = ym - h * dz / d
    # zs1 = zm - h * dy / d
    # zs2 = zm + h * dy / d

    # K1 = np.array([c_c[0], ys2, zs2])
    return p0


pos = np.array([20, 20, -170])

normal_vector = getAnglePlatform(pos[0], pos[1])
print "The normal vector is", normal_vector

new_p1, new_p2, new_p3 = getPlatformPositions(normal_vector, pos)

arm = 110.0
leg = 266.7
# length from center of base to axis of rotation
base_radius = 55

# define the position where the arms meet the base
base_1 = np.array([base_radius * np.cos(np.deg2rad(90)), base_radius * np.sin(np.deg2rad(90)), 0])
print "base1: ", base_1
base_2 = np.array([base_radius * np.cos(np.deg2rad(210)), base_radius * np.sin(np.deg2rad(210)), 0])
base_3 = np.array([base_radius * np.cos(np.deg2rad(330)), base_radius * np.sin(np.deg2rad(330)), 0])
# draw lines to platform
arm_link_1 = curve(pos=[(base_1), (base_2)])
arm_link_2 = curve(pos=[(base_2), (base_3)])
arm_link_3 = curve(pos=[(base_3), (base_1)])


K1 = getKneePosition(base_1, new_p1)
print "Knee1: ", K1

K2 = getKneePosition(base_2, new_p2)
print "Knee1: ", K2

K3 = getKneePosition(base_3, new_p3)
print "Knee1: ", K3

# Sphere Circle Intersection
# http://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection

# c_c = base_1
# print "c_c", c_c
# c_s = new_p1
# print "c_s", c_s
# n = np.array([1,0,0]) # normal vector to circle
# print "n: ", n
#
# d = np.dot(c_c - c_s, n)
# print "d: ", d
#
# if abs(d) > leg:
#     print "No intersection"
#
# if d == leg:
#     print "Only one intersection"
#
# c_p = c_s + (d * n)     # center of sphere circle cut by plane of intersection
# print "c_p", c_p
#
# r_p = np.sqrt(leg ** 2 - d ** 2)    # radius of sphere circle cut by plane
# print "r_p: ", r_p
#
# d_p = np.sqrt((c_c[0] - c_p[0])**2 + (c_c[1] - c_p[1])**2 + (c_c[2] - c_p[2])**2)
# print "d_p: ", d_p
#
# # solve for line of intersection between circles, find the center and radius
# # h = 0.5 + (arm**2 - leg**2) / (d_p**2)
# # a = (arm**2 - leg**2 + d_p**2) / (2 * d_p)
# # h = np.sqrt(leg**2 - a**2)
# # print "h", h
#
# dy = c_p[1] - c_c[1]
# dz = c_p[2] - c_c[2]
# d = np.sqrt(dy*dy+dz*dz)
#
# a = (arm*arm-r_p*r_p+d_p*d_p)/(2*d_p)
# h = np.sqrt(arm*arm-a*a)
# ym = c_c[1] + a*dy/d
# zm = c_c[2] + a*dz/d
# ys1 = ym + h*dz/d
# ys2 = ym - h*dz/d
# zs1 = zm - h*dy/d
# zs2 = zm + h*dy/d
#
# K1 = np.array([c_c[0], ys2, zs2])
# print K1

platform_k1 = curve(pos=[(K1), (new_p1)])
base_k1 = curve(pos=[(K1), (base_1)])
platform_k2 = curve(pos=[(K2), (new_p2)])
base_k2 = curve(pos=[(K2), (base_2)])
platform_k3 = curve(pos=[(K3), (new_p3)])
base_k3 = curve(pos=[(K3), (base_3)])

if arm != euclidean_distance(K1, base_1) or arm != euclidean_distance(K2, base_2) or arm != euclidean_distance(K3, base_3):
    print "ERROR: The arm length is incorrect"
    print "arm1 dist: ", euclidean_distance(K1, base_1)
    print "arm2 dist: ", euclidean_distance(K2, base_2)
    print "arm2 dist: ", euclidean_distance(K3, base_3)

if arm != euclidean_distance(K1, base_1):
    print "ERROR: The arm length is incorrect"
    print "arm1 dist: ", euclidean_distance(K1, base_1)

if arm != euclidean_distance(K2, base_2):
    print "ERROR: The arm length is incorrect"
    print "arm2 dist: ", euclidean_distance(K2, base_2)

if arm != euclidean_distance(K3, base_3):
    print "ERROR: The arm length is incorrect"
    print "arm2 dist: ", euclidean_distance(K3, base_3)

if leg != euclidean_distance(K1, new_p1) or leg != euclidean_distance(K2, new_p2) or leg != euclidean_distance(K3, new_p3):
    print "ERROR: The leg length is incorrect"
    print "leg1 dist: ", euclidean_distance(K1, new_p1)
    print "leg2 dist: ", euclidean_distance(K2, new_p2)
    print "leg2 dist: ", euclidean_distance(K3, new_p3)





