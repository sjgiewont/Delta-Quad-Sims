import numpy as np
from visual import *

# def angle_xy(base_1, new_p1, new_p1_cross, arm, leg):
#     CM = base_1 - new_p1
#
#     MM_1 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[0]
#     MM_2 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[1]
#     MM_3 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[2]
#
#     MM = np.sqrt((MM_1 ** 2) + (MM_2 ** 2) + (MM_3 ** 2))
#
#     CM_prime_1 = CM[0] - MM_1
#     CM_prime_2 = CM[1] - MM_2
#     CM_prime_3 = CM[2] - MM_3
#     CM_prime = np.sqrt((CM_prime_1 ** 2) + (CM_prime_2 ** 2) + (CM_prime_3 ** 2))
#
#     r_projection = np.sqrt((arm ** 2) - (MM ** 2))
#
#     i_x = CM_prime_1 / CM_prime
#     i_y = CM_prime_2 / CM_prime
#     i_z = CM_prime_3 / CM_prime
#
#     j_x = (new_p1_cross[1] * i_z) - (new_p1_cross[2] * i_y)
#     j_y = (new_p1_cross[2] * i_x) - (new_p1_cross[0] * i_z)
#     j_z = (new_p1_cross[0] * i_y) - (new_p1_cross[1] * i_x)
#
#     x_u_intersect = ((CM_prime ** 2) - (r_projection ** 2) + (leg ** 2)) / (2 * CM_prime)
#     y_u_intersect = np.sqrt((leg ** 2) - (x_u_intersect ** 2))
#
#     x_intersect = new_p1[0] + (i_x * x_u_intersect) - (j_x * y_u_intersect)
#     y_intersect = new_p1[1] + (i_y * x_u_intersect) - (j_y * y_u_intersect)
#     z_intersect = new_p1[2] + (i_z * x_u_intersect) - (j_z * y_u_intersect)
#
#     K1 = np.array([x_intersect, y_intersect, z_intersect])
#     return K1

def angle_xy(base_1, new_p1, new_p, arm, leg):
    e = 25     # platform
    f = 55    # base
    re = arm
    rf = leg
    # re = 52.4       #leg length
    # rf = 17.145     #arm length

    # y1 = -0.5 * 0.57735 * f
    y1 = base_1
    # y0 = y0 - (0.5 * 0.57735 * e)
    y0 = new_p1
    x0 = new_p[0]
    z0 = new_p[2]

    anp = ((x0 * x0) + (y0 * y0) + (z0 * z0) + (rf * rf))
    ann = ((re * re) + (y1 * y1))
    an = anp - ann
    ad = 2 * z0
    a = an / ad
    b = (y1 - y0) / z0
    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    print("d={0} y1={1} y0={2} a={3} b={4} an={5} ad={6} anp={7} ann={8}".format(d,y1,y0,a,b,an,ad,anp,ann))

    try:
        yj = (y1 - a * b - sqrt(d)) / (b * b + 1)
        zj = a + b * yj

        if yj > y1:
            angle = 180
        else:
            angle = 0

        return atan(-zj / (y1 - yj)) * 180 / pi + angle

    except:
        return

""" Returns the unit vector of the vector.  """
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

arm = 100
leg = 180
foot_height = 20

x0 = 0
y0 = 0
z0 = 0

P_x = 20
P_y = -20
P_z = -130

dist_platform_foot = np.sqrt((P_x - x0)**2 + (P_y - y0)**2 + (P_z - z0)**2)

# main_vector = np.array([P_x-x0, P_y - y0, P_z - z0])
# main_vector = np.array([x0-P_x, y0-P_y, z0-P_z])
# print "main_vector", main_vector


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


# length from center of platform (foot) to axis of rotation
foot_radius = 25
# define the position where the arms meet the base
platform_1 = np.array([foot_radius * np.cos(np.deg2rad(90)), foot_radius * np.sin(np.deg2rad(90)), -dist_platform_foot + foot_height])
platform_2 = np.array([foot_radius * np.cos(np.deg2rad(210)), foot_radius * np.sin(np.deg2rad(210)), -dist_platform_foot + foot_height])
platform_3 = np.array([foot_radius * np.cos(np.deg2rad(330)), foot_radius * np.sin(np.deg2rad(330)), -dist_platform_foot + foot_height])

# draw lines to show platform
# arm_link_1 = curve(pos=[(platform_1), (platform_2)])
# arm_link_2 = curve(pos=[(platform_2), (platform_3)])
# arm_link_3 = curve(pos=[(platform_3), (platform_1)])

print "base_p1", base_1
p = np.matrix([[0], [0], [-dist_platform_foot + foot_height], [1]])
p1 = np.matrix([[platform_1[0]], [platform_1[1]], [platform_1[2]], [1]])
p2 = np.matrix([[platform_2[0]], [platform_2[1]], [platform_2[2]], [1]])
p3 = np.matrix([[platform_3[0]], [platform_3[1]], [platform_3[2]], [1]])

alpha = 90 + np.degrees(np.arctan2(P_z, P_y))
beta = 90 + np.degrees(np.arctan2(P_z, P_x))
gamma = 0

rotate_x = np.matrix([[1, 0, 0, 0], [0, np.cos(np.radians(alpha)), -np.sin(np.radians(alpha)), 0], [0, np.sin(np.radians(alpha)), np.cos(np.radians(alpha)), 0], [0, 0, 0, 1]])
rotate_y = np.matrix([[np.cos(np.radians(beta)), 0, np.sin(np.radians(beta)), 0], [0, 1, 0 , 0], [-np.sin(np.radians(beta)), 0, np.cos(np.radians(beta)), 0], [0, 0, 0, 1]])
rotate_z = np.matrix([[np.cos(np.radians(gamma)), -np.sin(np.radians(gamma)), 0, 0], [np.sin(np.radians(gamma)), np.cos(np.radians(gamma)), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

new_p = rotate_y * rotate_x * p
new_p1 = rotate_y * rotate_x * p1
new_p2 = rotate_y * rotate_x * p2
new_p3 = rotate_y * rotate_x * p3

print "new_p1", new_p1
new_p = np.array([new_p[0, 0], new_p[1, 0], new_p[2, 0]])
main_vector_line = curve(pos=[(new_p), ([0, 0, 0])])

new_p1 = np.array([new_p1[0, 0], new_p1[1, 0], new_p1[2, 0]])
new_p2 = np.array([new_p2[0, 0], new_p2[1, 0], new_p2[2, 0]])
new_p3 = np.array([new_p3[0, 0], new_p3[1, 0], new_p3[2, 0]])

print "newp1", new_p1

# draw lines to show arms
arm_link_4 = curve(pos=[(new_p1), (new_p2)])
arm_link_5 = curve(pos=[(new_p2), (new_p3)])
arm_link_6 = curve(pos=[(new_p3), (new_p1)])

new_p1_v = np.array(new_p1 - new_p)
print "new_p1_v", new_p1_v
new_p2_v = np.array(new_p2 - new_p)
new_p3_v = np.array(new_p3 - new_p)
platform_v1 = curve(pos=[(new_p), (new_p1)])
platform_v2 = curve(pos=[(new_p), (new_p2)])
platform_v3 = curve(pos=[(new_p), (new_p3)])

main_vector = np.array([x0-new_p[0], y0-new_p[1], z0-new_p[2]])
print "main_vector", main_vector

main_vector = main_vector/np.linalg.norm(main_vector)
new_p1_v = new_p1_v/np.linalg.norm(new_p1_v)
new_p2_v = new_p2_v/np.linalg.norm(new_p2_v)
new_p3_v = new_p3_v/np.linalg.norm(new_p3_v)

new_p1_cross = np.cross(main_vector, new_p1_v)
new_p2_cross = np.cross(main_vector, new_p2_v)
new_p3_cross = np.cross(main_vector, new_p3_v)

print "new_p1_cross", new_p1_cross

platform_cross1 = curve(pos=[(new_p1 + new_p1_cross), (new_p1)])
platform_cross2 = curve(pos=[(new_p2 + new_p2_cross), (new_p2)])
platform_cross3 = curve(pos=[(new_p3 + new_p3_cross), (new_p3)])


#if platform along x axis only
# return self.circle_intersection_sympy(circle1,circle2)
# x1, y1, r1 = circle1
# x2, y2, r2 = circle2
if P_x == 0:
    x1 = base_1[1]
    y1 = base_1[2]
    r1 = arm

    x2 = new_p1[1]
    y2 = new_p1[2]
    r2 = leg

    # http://stackoverflow.com/a/3349134/798588
    dx, dy = x2 - x1, y2 - y1
    d = np.sqrt(dx * dx + dy * dy)
    print "d: ", d
    if d > r1 + r2:
        print "#1"
    if d < abs(r1 - r2):
        print "#2"
    if d == 0 and r1 == r2:
        print "#3"

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = np.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / d
    ym = y1 + a * dy / d
    xs1 = xm + h * dy / d
    xs2 = xm - h * dy / d
    ys1 = ym - h * dx / d
    ys2 = ym + h * dx / d

    K1 = np.array([0, xs1, ys1])
    print "K1: ", K1
    K2 = np.array([0, xs2, ys2])

    leg_1 = curve(pos=[(new_p1), (K2)])
    arm_1 = curve(pos=[(base_1), (K2)])

    leg_dist = np.sqrt((K2[0]-new_p1[0])**2 + (K2[1]-new_p1[1])**2 + (K2[2]-new_p1[2])**2)
    print leg_dist

    arm_dist = np.sqrt((K2[0]-base_1[0])**2 + (K2[1]-base_1[1])**2 + (K2[2]-base_1[2])**2)
    print arm_dist


line_dir = np.dot(new_p1, new_p1_cross)
print "line dir", line_dir


a = (line_dir / new_p1_cross[1])
b = -(new_p1_cross[2] / new_p1_cross[1])

A = (b ** 2) + 1
B = (2 * a * b) - (2 * b * new_p1[1]) - (2 * new_p1[2])
C = (new_p1[0] ** 2) + (a ** 2) - (2 * a * new_p1[1]) + (new_p1[1] ** 2) + (new_p1[2] ** 2) - (leg ** 2)

print a, b
print A, B, C

z = (-B - np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
print "z", z

y = a + (b * z)
print "y", y

K1 = np.array([0, y, z])

platform_k1 = curve(pos=[(K1), (new_p1)])
base_k1 = curve(pos=[(K1), (base_1)])

leg_dist = np.sqrt((K1[0]-new_p1[0])**2 + (K1[1]-new_p1[1])**2 + (K1[2]-new_p1[2])**2)
print leg_dist

arm_dist = np.sqrt((K1[0]-base_1[0])**2 + (K1[1]-base_1[1])**2 + (K1[2]-base_1[2])**2)
print arm_dist

# N = new_p1_cross
# U = np.array([0, arm, 0])
# V = np.array([0, 0, arm])
# C0 = base_1
# P = new_p1
#
# a = np.dot(N, U)
# b = np.dot(N, V)
# c = -np.dot(N, C0-P)
#
# print a, b, c
# arg1 = c / (np.sqrt((a**2) + (b**2)))
# arg2 = a / (np.sqrt((a**2) + (b**2)))
# print "args", arg1, arg2
#
# print  np.arccos(arg2)
# t = np.arcsin(arg1) - np.arccos(arg2)
#
# print "t", t


# c = ((2 * new_p1_cross[0] * new_p1[0] + new_p1_cross[1] * new_p1[1] + new_p1_cross[2] * new_p1[2]) / new_p1_cross[2]) + new_p1[2] + ((new_p1_cross[1] * new_p1[1]) / new_p1_cross[2])
#
# m = -(new_p1_cross[1] / new_p1_cross[2])
# print "m", m
#
# p = base_1[1]
# q = base_1[2]
#
# B = 2 * (m * c - m * q - p)
# A = m ** 2 + 1
# C = (q ** 2) - (arm ** 2) + (p ** 2) - (2 * c * q) + (c ** 2)
#
# y = (-B + np.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
# z = (m * y) + c
# print y, z

# cdiff = (base_1 - new_p1)
# cdifflen = np.sqrt((base_1[0]-new_p1[0])**2 + (base_1[1]-new_p1[1])**2 + (base_1[2]-new_p1[2])**2)
#
# cdiffnorm = unit_vector(cdiff)
#
# base_1_cross = np.cross(base_1, [0, 0, 1])
#
# cdiffperp = cdiffnorm * unit_vector(base_1_cross)
#
# q = (cdifflen ** 2) + (leg ** 2) - (arm ** 2)
# dx = (0.5 * q) / cdifflen
# dy = (0.5 * np.sqrt(4 * (cdifflen**2) * (leg**2) - (q**2))) / cdifflen
#
# P1 = base_1 + cdiffnorm * dx + cdiffperp * dy
# print "P!", P1
# test = curve(pos=[(P1), (base_1)])


# # inputs: base_1, new_p1, new_p1_cross, arm, leg
#
# CM = base_1 - new_p1
#
# MM_1 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[0]
# MM_2 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[1]
# MM_3 = ((new_p1_cross[0] * CM[0]) + (new_p1_cross[1] * CM[1]) + (new_p1_cross[2] * CM[2])) * new_p1_cross[2]
#
# MM = np.sqrt((MM_1 ** 2) + (MM_2 ** 2) + (MM_3 ** 2))
#
# CM_prime_1 = CM[0] - MM_1
# CM_prime_2 = CM[1] - MM_2
# CM_prime_3 = CM[2] - MM_3
# CM_prime = np.sqrt((CM_prime_1 ** 2) + (CM_prime_2 ** 2) + (CM_prime_3 ** 2))
#
# r_projection = np.sqrt((arm ** 2) - (MM ** 2))
#
# i_x = CM_prime_1 / CM_prime
# i_y = CM_prime_2 / CM_prime
# i_z = CM_prime_3 / CM_prime
#
# j_x = (new_p1_cross[1] * i_z) - (new_p1_cross[2] * i_y)
# j_y = (new_p1_cross[2] * i_x) - (new_p1_cross[0] * i_z)
# j_z = (new_p1_cross[0] * i_y) - (new_p1_cross[1] * i_x)
#
# x_u_intersect = ((CM_prime ** 2) - (r_projection ** 2) + (leg ** 2)) / (2 * CM_prime)
# y_u_intersect = np.sqrt((leg ** 2) - (x_u_intersect ** 2))
#
# x_intersect = new_p1[0] + (i_x * x_u_intersect) - (j_x * y_u_intersect)
# y_intersect = new_p1[1] + (i_y * x_u_intersect) - (j_y * y_u_intersect)
# z_intersect = new_p1[2] + (i_z * x_u_intersect) - (j_z * y_u_intersect)
#
# K1 = np.array([x_intersect, y_intersect, z_intersect])

# K1 = angle_xy(base_1, new_p1, new_p1_cross, arm, leg)
# K2 = angle_xy(base_2, new_p2, new_p2_cross, arm, leg)
# K3 = angle_xy(base_3, new_p3, new_p3_cross, arm, leg)
#
# theta_1 = angle_xy(base_1, new_p1, new_p, arm, leg)
# print theta_1

# platform_k1 = curve(pos=[(K1), (new_p1)])
# base_k1 = curve(pos=[(K1), (base_1)])
# platform_k2 = curve(pos=[(K2), (new_p2)])
# base_k2 = curve(pos=[(K2), (base_2)])
# platform_k3 = curve(pos=[(K3), (new_p3)])
# base_k3 = curve(pos=[(K3), (base_3)])
#
# arm_v1 = base_1 - K1
# print "the angle is", angle_between(base_1, arm_v1)
#
# leg_dist = np.sqrt((K1[0]-new_p1[0])**2 + (K1[1]-new_p1[1])**2 + (K1[2]-new_p1[2])**2)
# print leg_dist
#
# arm_dist = np.sqrt((K1[0]-base_1[0])**2 + (K1[1]-base_1[1])**2 + (K1[2]-base_1[2])**2)
# print arm_dist


# print "base_1", base_1
# print "normal", new_p1_cross
# print "new P1", new_p1

#
# # the first knee, in the universal y,z plane, pointing towards 12 o'clock
# k1 = platform_1 + np.array([0, -arm*cos(np.deg2rad(theta1)), arm*sin(np.deg2rad(theta1))])
#
# # second knee, pointing near 7 o'clock
# k2 = platform_2 + np.array([0, -arm*cos(np.deg2rad(theta2)), arm*sin(np.deg2rad(theta2))])
# k2 = np.array([k2[0]*cos(np.deg2rad(120)) - k2[1]*sin(np.deg2rad(120)) + platform_2[0] - cos(np.deg2rad(120))*platform_2[0] + sin(np.deg2rad(120))*platform_2[1], k2[0]*sin(np.deg2rad(120)) + k2[1]*cos(np.deg2rad(120)) + platform_2[1] - sin(np.deg2rad(120))*platform_2[0] - cos(np.deg2rad(120))*platform_2[1], k2[2]])
#
# # third knee, pointing near 5 o'clock
# k3 = base_p3 + np.array([0, -arm*cos(np.deg2rad(theta3)), arm*sin(np.deg2rad(theta3))])
# k3 = np.array([k3[0]*cos(np.deg2rad(240)) - k3[1]*sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240))*base_p3[0] + sin(np.deg2rad(240))*base_p3[1], k3[0]*sin(np.deg2rad(240)) + k3[1]*cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240))*base_p3[0] - cos(np.deg2rad(240))*base_p3[1], k3[2]])
#
# arm_link_1 = curve(pos=[(platform_1), (k1)])
#
# leg_dist = np.sqrt((k1[0]-new_p1[0])**2 + (k1[1]-new_p1[1])**2 + (k1[2]-new_p1[2])**2)
# if leg_dist != leg:
#     print "Error Leg distance is: ", leg_dist
# leg_link_1 = curve(pos=[(k1), (new_p1)])

