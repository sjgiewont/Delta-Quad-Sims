from visual import *
import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm
import time
from anfis import *
import timeit
import cPickle as pickle

# Find the intersection of three spheres
# P1,P2,P3 are the centers, r1,r2,r3 are the radii
# Implementaton based on Wikipedia Trilateration article.
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

def vPython_user_rotate():
    xmax = 60
    scene.range = xmax
    initialdistance = xmax / tan(scene.fov / 2.0)

    # Wait while user adjusts the view.
    scene.mouse.getclick()

    # Determine how far we are from the center of the scene.
    savedistance = mag(scene.mouse.camera - scene.center)
    # Keep a copy of current scene.forward.
    saveforward = vector(scene.forward)
    ### Mousing changes scene.forward and can change scene.up too!
    saveup = vector(scene.up)

    # Wait for more mousing, but on the next click, zoom out programatically:
    scene.mouse.getclick()
    # Changing scene.range affects the view (but not vice versa).
    scene.range = 2 * scene.range
    savedistance *= scene.range.x / xmax

    # Wait while user changes the view, then restore the saved view.
    scene.mouse.getclick()
    # Determine the new distance from the center.
    distance = mag(scene.mouse.camera - scene.center)
    # Adjust the range based on the new distance.
    scene.range = (savedistance / distance) * xmax
    # Reset the viewing direction.
    scene.up = saveup
    scene.forward = saveforward


def forwardKinematics(theta_1, theta_2, theta_3):
    # length from center of base to axis of rotation
    base_radius = 55
    # length from center of platform (foot) to axis of rotation
    foot_radius = 25
    # length of arm and leg
    arm = 100
    leg = 200

    # define the position where the arms meet the base
    base_p1 = np.array([base_radius * cos(np.deg2rad(90)), base_radius * sin(np.deg2rad(90)), 0])
    base_p2 = np.array([base_radius * cos(np.deg2rad(210)), base_radius * sin(np.deg2rad(210)), 0])
    base_p3 = np.array([base_radius * cos(np.deg2rad(330)), base_radius * sin(np.deg2rad(330)), 0])

    # the first knee, in the universal y,z plane, pointing towards 12 o'clock
    k1 = base_p1 + np.array([0, -arm * cos(np.deg2rad(theta_1)), arm * sin(np.deg2rad(theta_1))])

    # second knee, pointing near 7 o'clock
    k2 = base_p2 + np.array([0, -arm * cos(np.deg2rad(theta_2)), arm * sin(np.deg2rad(theta_2))])
    k2 = np.array([k2[0] * cos(np.deg2rad(120)) - k2[1] * sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120)) * base_p2[0] + sin(np.deg2rad(120)) * base_p2[1], k2[0] * sin(np.deg2rad(120)) + k2[1] * cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120)) * base_p2[0] - cos(np.deg2rad(120)) * base_p2[1], k2[2]])

    # third knee, pointing near 5 o'clock
    k3 = base_p3 + np.array([0, -arm * cos(np.deg2rad(theta_3)), arm * sin(np.deg2rad(theta_3))])
    k3 = np.array([k3[0] * cos(np.deg2rad(240)) - k3[1] * sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240)) * base_p3[0] + sin(np.deg2rad(240)) * base_p3[1], k3[0] * sin(np.deg2rad(240)) + k3[1] * cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240)) * base_p3[0] - cos(np.deg2rad(240)) * base_p3[1], k3[2]])

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

    foot_pt_x = (circle_intersect_2[0] + vx * (height_intersect_2_platform))
    foot_pt_y = (circle_intersect_2[1] + vy * (height_intersect_2_platform))
    foot_pt_z = (circle_intersect_2[2] + vz * (height_intersect_2_platform))

    foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z])

    return foot_pt


def setupAnimation():
    global arm_link_1
    global arm_link_2
    global arm_link_3
    global leg_link_1
    global leg_link_2
    global leg_link_3
    global foot_1

    # the angle each of the arms are oriented. 180 deg is horizontal
    theta_1 = 240
    theta_2 = 240
    theta_3 = 240

    # length from center of base to axis of rotation
    base_radius = 55
    # length from center of platform (foot) to axis of rotation
    foot_radius = 25
    # length of arm and leg
    arm = 100
    leg = 200

    # define the position where the arms meet the base
    base_p1 = np.array([base_radius * cos(np.deg2rad(90)), base_radius * sin(np.deg2rad(90)), 0])
    base_p2 = np.array([base_radius * cos(np.deg2rad(210)), base_radius * sin(np.deg2rad(210)), 0])
    base_p3 = np.array([base_radius * cos(np.deg2rad(330)), base_radius * sin(np.deg2rad(330)), 0])

    # draw the pivot points of the base
    ball = sphere(pos=(base_p1), radius=10, material=materials.rough)
    ball = sphere(pos=(base_p2), radius=10, material=materials.rough)
    ball = sphere(pos=(base_p3), radius=10, material=materials.rough)

    # the first knee, in the universal y,z plane, pointing towards 12 o'clock
    k1 = base_p1 + np.array([0, -arm * cos(np.deg2rad(theta_1)), arm * sin(np.deg2rad(theta_1))])

    # second knee, pointing near 7 o'clock
    k2 = base_p2 + np.array([0, -arm * cos(np.deg2rad(theta_2)), arm * sin(np.deg2rad(theta_2))])
    k2 = np.array([k2[0] * cos(np.deg2rad(120)) - k2[1] * sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120)) *
                   base_p2[0] + sin(np.deg2rad(120)) * base_p2[1],
                   k2[0] * sin(np.deg2rad(120)) + k2[1] * cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120)) *
                   base_p2[0] - cos(np.deg2rad(120)) * base_p2[1], k2[2]])

    # third knee, pointing near 5 o'clock
    k3 = base_p3 + np.array([0, -arm * cos(np.deg2rad(theta_3)), arm * sin(np.deg2rad(theta_3))])
    k3 = np.array([k3[0] * cos(np.deg2rad(240)) - k3[1] * sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240)) *
                   base_p3[0] + sin(np.deg2rad(240)) * base_p3[1],
                   k3[0] * sin(np.deg2rad(240)) + k3[1] * cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240)) *
                   base_p3[0] - cos(np.deg2rad(240)) * base_p3[1], k3[2]])

    # distances between knees
    dist_k1_k2 = np.linalg.norm(k1 - k2)
    dist_k1_k3 = np.linalg.norm(k1 - k3)
    dist_k2_k3 = np.linalg.norm(k2 - k3)

    # draw lines to show arms
    arm_link_1 = curve(pos=[(base_p1), (k1)])
    arm_link_2 = curve(pos=[(base_p2), (k2)])
    arm_link_3 = curve(pos=[(base_p3), (k3)])

    # find the intersection of the three arm spheres
    circle_intersect_1, circle_intersect_2 = trilaterate(k1, k2, k3, leg, leg, leg)

    # draw lines to show arms
    leg_link_1 = curve(pos=[(k1), (circle_intersect_2)])
    leg_link_2 = curve(pos=[(k2), (circle_intersect_2)])
    leg_link_3 = curve(pos=[(k3), (circle_intersect_2)])


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

    vmag = np.sqrt(
        vx_circle_intersect * vx_circle_intersect + vy_circle_intersect * vy_circle_intersect + vz_circle_intersect * vz_circle_intersect)

    vx = vx_circle_intersect / vmag
    vy = vy_circle_intersect / vmag
    vz = vz_circle_intersect / vmag

    foot_pt_x = (circle_intersect_2[0] + vx * (height_intersect_2_platform))
    foot_pt_y = (circle_intersect_2[1] + vy * (height_intersect_2_platform))
    foot_pt_z = (circle_intersect_2[2] + vz * (height_intersect_2_platform))

    foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z])

    foot_1 = sphere(pos=(foot_pt), radius=5, material=materials.rough)

    scene.up = vector(1, 0, 0)
    # scene.up = scene.up.rotate(angle=1.57, axis=scene.forward)
    scene.forward = scene.forward.rotate(angle=1.57, axis=scene.up)
    # scene.forward = vector(1, 0, 0)
    scene.up = scene.up.rotate(angle=-1.57, axis=scene.forward)

    rate(20)

    scene.range = 275
    scene.center
    scene.autocenter

def displayPosition(theta_1, theta_2, theta_3):
    global arm_link_1
    global arm_link_2
    global arm_link_3
    global leg_link_1
    global leg_link_2
    global leg_link_3
    global foot_1
    # length from center of base to axis of rotation
    base_radius = 55
    # length from center of platform (foot) to axis of rotation
    foot_radius = 25
    # length of arm and leg
    arm = 100
    leg = 200

    # define the position where the arms meet the base
    base_p1 = np.array([base_radius * cos(np.deg2rad(90)), base_radius * sin(np.deg2rad(90)), 0])
    base_p2 = np.array([base_radius * cos(np.deg2rad(210)), base_radius * sin(np.deg2rad(210)), 0])
    base_p3 = np.array([base_radius * cos(np.deg2rad(330)), base_radius * sin(np.deg2rad(330)), 0])

    # draw the pivot points of the base
    ball = sphere(pos=(base_p1), radius=10, material=materials.rough)
    ball = sphere(pos=(base_p2), radius=10, material=materials.rough)
    ball = sphere(pos=(base_p3), radius=10, material=materials.rough)

    # the first knee, in the universal y,z plane, pointing towards 12 o'clock
    k1 = base_p1 + np.array([0, -arm * cos(np.deg2rad(theta_1)), arm * sin(np.deg2rad(theta_1))])

    # second knee, pointing near 7 o'clock
    k2 = base_p2 + np.array([0, -arm * cos(np.deg2rad(theta_2)), arm * sin(np.deg2rad(theta_2))])
    k2 = np.array([k2[0] * cos(np.deg2rad(120)) - k2[1] * sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120)) *
                   base_p2[0] + sin(np.deg2rad(120)) * base_p2[1],
                   k2[0] * sin(np.deg2rad(120)) + k2[1] * cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120)) *
                   base_p2[0] - cos(np.deg2rad(120)) * base_p2[1], k2[2]])

    # third knee, pointing near 5 o'clock
    k3 = base_p3 + np.array([0, -arm * cos(np.deg2rad(theta_3)), arm * sin(np.deg2rad(theta_3))])
    k3 = np.array([k3[0] * cos(np.deg2rad(240)) - k3[1] * sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240)) *
                   base_p3[0] + sin(np.deg2rad(240)) * base_p3[1],
                   k3[0] * sin(np.deg2rad(240)) + k3[1] * cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240)) *
                   base_p3[0] - cos(np.deg2rad(240)) * base_p3[1], k3[2]])

    # distances between knees
    dist_k1_k2 = np.linalg.norm(k1 - k2)
    dist_k1_k3 = np.linalg.norm(k1 - k3)
    dist_k2_k3 = np.linalg.norm(k2 - k3)

    # draw lines to show arms
    arm_link_1.visible = False
    arm_link_2.visible = False
    arm_link_3.visible = False
    arm_link_1 = curve(pos=[(base_p1), (k1)])
    arm_link_2 = curve(pos=[(base_p2), (k2)])
    arm_link_3 = curve(pos=[(base_p3), (k3)])


    # find the intersection of the three arm spheres
    circle_intersect_1, circle_intersect_2 = trilaterate(k1, k2, k3, leg, leg, leg)

    # draw lines to show arms
    leg_link_1.visible = False
    leg_link_2.visible = False
    leg_link_3.visible = False
    leg_link_1 = curve(pos=[(k1), (circle_intersect_2)])
    leg_link_2 = curve(pos=[(k2), (circle_intersect_2)])
    leg_link_3 = curve(pos=[(k3), (circle_intersect_2)])


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

    vmag = np.sqrt(
        vx_circle_intersect * vx_circle_intersect + vy_circle_intersect * vy_circle_intersect + vz_circle_intersect * vz_circle_intersect)

    vx = vx_circle_intersect / vmag
    vy = vy_circle_intersect / vmag
    vz = vz_circle_intersect / vmag

    foot_pt_x = (circle_intersect_2[0] + vx * (height_intersect_2_platform))
    foot_pt_y = (circle_intersect_2[1] + vy * (height_intersect_2_platform))
    foot_pt_z = (circle_intersect_2[2] + vz * (height_intersect_2_platform))

    foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z])

    foot_1.visible = False
    foot_1 = sphere(pos=(foot_pt), radius=5, material=materials.rough)
    #foot_1.pos = foot_pt

    rate(24)

def inverseKinematics(x, y, z, anf):
    # with open('fuzzycontrol.pkl', 'rb') as f:
    #     anf = pickle.load(f)

    var = np.array([[x, y, z]])

    # print the predicted value based on the trained set
    theta = anfis.predict(anf, var)
    return theta

angle = 0
setupAnimation()
with open('fuzzycontrol_150_210_3.pkl', 'rb') as f:
    anf = pickle.load(f)
while 1:
    angle += 0.1
    x = 100*np.sin(angle)
    theta = inverseKinematics(x, 0, -130, anf)

    displayPosition(theta[0][0], theta[0][1], theta[0][2])

# # length from center of base to axis of rotation
# base_radius = 55
# # length from center of platform (foot) to axis of rotation
# foot_radius = 25
# # length of arm and leg
# arm = 100
# leg = 200
#
# # define the position where the arms meet the base
# base_p1 = np.array([base_radius*cos(np.deg2rad(90)),base_radius*sin(np.deg2rad(90)),0])
# base_p2 = np.array([base_radius*cos(np.deg2rad(210)),base_radius*sin(np.deg2rad(210)),0])
# base_p3 = np.array([base_radius*cos(np.deg2rad(330)),base_radius*sin(np.deg2rad(330)),0])
#
#
# # draw the pivot points of the base
# ball = sphere(pos=(base_p1), radius=10,  material=materials.rough)
# ball = sphere(pos=(base_p2), radius=10, material=materials.rough)
# ball = sphere(pos=(base_p3), radius=10, material=materials.rough)
#
# # the angle each of the arms are oriented. 180 deg is horizontal
# theta_1 = 240
# theta_2 = 240
# theta_3 = 240
#
# # the first knee, in the universal y,z plane, pointing towards 12 o'clock
# k1 = base_p1 + np.array([0, -arm*cos(np.deg2rad(theta_1)), arm*sin(np.deg2rad(theta_1))])
#
# # second knee, pointing near 7 o'clock
# k2 = base_p2 + np.array([0, -arm*cos(np.deg2rad(theta_2)), arm*sin(np.deg2rad(theta_2))])
# k2 = np.array([k2[0]*cos(np.deg2rad(120)) - k2[1]*sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120))*base_p2[0] + sin(np.deg2rad(120))*base_p2[1], k2[0]*sin(np.deg2rad(120)) + k2[1]*cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120))*base_p2[0] - cos(np.deg2rad(120))*base_p2[1], k2[2]])
#
# # third knee, pointing near 5 o'clock
# k3 = base_p3 + np.array([0, -arm*cos(np.deg2rad(theta_3)), arm*sin(np.deg2rad(theta_3))])
# k3 = np.array([k3[0]*cos(np.deg2rad(240)) - k3[1]*sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240))*base_p3[0] + sin(np.deg2rad(240))*base_p3[1], k3[0]*sin(np.deg2rad(240)) + k3[1]*cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240))*base_p3[0] - cos(np.deg2rad(240))*base_p3[1], k3[2]])
#
# # distances between knees
# dist_k1_k2 = np.linalg.norm(k1-k2)
# dist_k1_k3 = np.linalg.norm(k1-k3)
# dist_k2_k3 = np.linalg.norm(k2-k3)
#
# # draw lines to show arms
# arm_link_1 = curve(pos=[(base_p1), (k1)])
# arm_link_2 = curve(pos=[(base_p2), (k2)])
# arm_link_3 = curve(pos=[(base_p3), (k3)])
#
# # find the intersection of the three arm spheres
# circle_intersect_1, circle_intersect_2 = trilaterate(k1, k2, k3, leg, leg, leg)
# print circle_intersect_1, circle_intersect_2
#
# # draw lines to show legs
# leg_link_1 = curve(pos=[(k1), (circle_intersect_2)])
# leg_link_2 = curve(pos=[(k2), (circle_intersect_2)])
# leg_link_3 = curve(pos=[(k3), (circle_intersect_2)])
#
# # find the distance between intersection point and the platform
# height_knee_2_intersect_1 = np.sqrt(leg*leg - 0.25*(dist_k1_k2*dist_k1_k2))
# height_knee_2_intersect_2 = np.sqrt(leg*leg - 0.25*(dist_k1_k3*dist_k1_k3))
# height_knee_2_intersect_3 = np.sqrt(leg*leg - 0.25*(dist_k2_k3*dist_k2_k3))
# height_knee_2_intersect = np.mean([height_knee_2_intersect_1, height_knee_2_intersect_2, height_knee_2_intersect_3])
#
# height_knee_2_platform_1 = np.sqrt(leg*leg - 0.25*(dist_k1_k2 - foot_radius)*(dist_k1_k2 - foot_radius))
# height_knee_2_platform_2 = np.sqrt(leg*leg - 0.25*(dist_k1_k3 - foot_radius)*(dist_k1_k3 - foot_radius))
# height_knee_2_platform_3 = np.sqrt(leg*leg - 0.25*(dist_k2_k3 - foot_radius)*(dist_k2_k3 - foot_radius))
# height_knee_2_platform = np.mean([height_knee_2_platform_1, height_knee_2_platform_2, height_knee_2_platform_3])
#
# height_intersect_2_platform = height_knee_2_platform - height_knee_2_intersect
# print height_intersect_2_platform
#
#
# # Using vectors, find point along a line, given a distance.
# vx_circle_intersect = circle_intersect_2[0] - circle_intersect_1[0]
# vy_circle_intersect = circle_intersect_2[1] - circle_intersect_1[1]
# vz_circle_intersect = circle_intersect_2[2] - circle_intersect_1[2]
# print vx_circle_intersect, vy_circle_intersect, vz_circle_intersect
#
# vmag = np.sqrt(vx_circle_intersect*vx_circle_intersect + vy_circle_intersect*vy_circle_intersect + vz_circle_intersect*vz_circle_intersect)
#
# vx = vx_circle_intersect / vmag
# vy = vy_circle_intersect / vmag
# vz = vz_circle_intersect / vmag
#
# foot_pt_x = (circle_intersect_2[0] + vx * (height_intersect_2_platform))
# foot_pt_y = (circle_intersect_2[1] + vy * (height_intersect_2_platform))
# foot_pt_z = (circle_intersect_2[2] + vz * (height_intersect_2_platform))
#
# foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z])
# print foot_pt
#
# foot_1 = sphere(pos=(foot_pt), radius=5, material=materials.rough)
#
# rate(10)
#
# scene.up = vector(1,0,0)
# # scene.up = scene.up.rotate(angle=1.57, axis=scene.forward)
# scene.forward = scene.forward.rotate(angle=1.57, axis=scene.up)
# # scene.forward = vector(1, 0, 0)
# scene.up = scene.up.rotate(angle=-1.57, axis=scene.forward)
#
# scene.range = 275
# scene.center
#
# scene.autocenter
# # scene.up = scene.up.rotate(angle=180, axis=scene.forward)
#
#
# while 1:
#
#     theta_1 -= 1
#     theta_2 -= 1
#     theta_3 -= 1
#     print theta_3
#
#     # the first knee, in the universal y,z plane, pointing towards 12 o'clock
#     k1 = base_p1 + np.array([0, -arm*cos(np.deg2rad(theta_1)), arm*sin(np.deg2rad(theta_1))])
#
#     # second knee, pointing near 7 o'clock
#     k2 = base_p2 + np.array([0, -arm*cos(np.deg2rad(theta_2)), arm*sin(np.deg2rad(theta_2))])
#     k2 = np.array([k2[0]*cos(np.deg2rad(120)) - k2[1]*sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120))*base_p2[0] + sin(np.deg2rad(120))*base_p2[1], k2[0]*sin(np.deg2rad(120)) + k2[1]*cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120))*base_p2[0] - cos(np.deg2rad(120))*base_p2[1], k2[2]])
#
#     # third knee, pointing near 5 o'clock
#     k3 = base_p3 + np.array([0, -arm*cos(np.deg2rad(theta_3)), arm*sin(np.deg2rad(theta_3))])
#     k3 = np.array([k3[0]*cos(np.deg2rad(240)) - k3[1]*sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240))*base_p3[0] + sin(np.deg2rad(240))*base_p3[1], k3[0]*sin(np.deg2rad(240)) + k3[1]*cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240))*base_p3[0] - cos(np.deg2rad(240))*base_p3[1], k3[2]])
#
#     # distances between knees
#     dist_k1_k2 = np.linalg.norm(k1-k2)
#     dist_k1_k3 = np.linalg.norm(k1-k3)
#     dist_k2_k3 = np.linalg.norm(k2-k3)
#
#     # draw lines to show arms
#     arm_link_1.visible = False
#     arm_link_2.visible = False
#     arm_link_3.visible = False
#     arm_link_1 = curve(pos=[(base_p1), (k1)])
#     arm_link_2 = curve(pos=[(base_p2), (k2)])
#     arm_link_3 = curve(pos=[(base_p3), (k3)])
#
#     # arm_link_1 = curve(pos=[(base_p1), (k1)])
#     # arm_link_2 = curve(pos=[(base_p2), (k2)])
#     # arm_link_3 = curve(pos=[(base_p3), (k3)])
#
#     # find the intersection of the three arm spheres
#     circle_intersect_1, circle_intersect_2 = trilaterate(k1, k2, k3, leg, leg, leg)
#     print circle_intersect_1, circle_intersect_2
#
#     # draw lines to show arms
#     leg_link_1.visible = False
#     leg_link_2.visible = False
#     leg_link_3.visible = False
#
#     leg_link_1 = curve(pos=[(k1), (circle_intersect_2)])
#     leg_link_2 = curve(pos=[(k2), (circle_intersect_2)])
#     leg_link_3 = curve(pos=[(k3), (circle_intersect_2)])
#
#     # arm_link_1 = curve(pos=[(k1), (circle_intersect_2)])
#     # arm_link_2 = curve(pos=[(k2), (circle_intersect_2)])
#     # arm_link_3 = curve(pos=[(k3), (circle_intersect_2)])
#
#     # find the distance between intersection point and the platform
#     height_knee_2_intersect_1 = np.sqrt(leg*leg - 0.25*(dist_k1_k2*dist_k1_k2))
#     height_knee_2_intersect_2 = np.sqrt(leg*leg - 0.25*(dist_k1_k3*dist_k1_k3))
#     height_knee_2_intersect_3 = np.sqrt(leg*leg - 0.25*(dist_k2_k3*dist_k2_k3))
#     height_knee_2_intersect = np.mean([height_knee_2_intersect_1, height_knee_2_intersect_2, height_knee_2_intersect_3])
#
#     height_knee_2_platform_1 = np.sqrt(leg*leg - 0.25*(dist_k1_k2 - foot_radius)*(dist_k1_k2 - foot_radius))
#     height_knee_2_platform_2 = np.sqrt(leg*leg - 0.25*(dist_k1_k3 - foot_radius)*(dist_k1_k3 - foot_radius))
#     height_knee_2_platform_3 = np.sqrt(leg*leg - 0.25*(dist_k2_k3 - foot_radius)*(dist_k2_k3 - foot_radius))
#     height_knee_2_platform = np.mean([height_knee_2_platform_1, height_knee_2_platform_2, height_knee_2_platform_3])
#
#     height_intersect_2_platform = height_knee_2_platform - height_knee_2_intersect
#     print height_intersect_2_platform
#
#
#     # Using vectors, find point along a line, given a distance.
#     vx_circle_intersect = circle_intersect_2[0] - circle_intersect_1[0]
#     vy_circle_intersect = circle_intersect_2[1] - circle_intersect_1[1]
#     vz_circle_intersect = circle_intersect_2[2] - circle_intersect_1[2]
#     print vx_circle_intersect, vy_circle_intersect, vz_circle_intersect
#
#     vmag = np.sqrt(vx_circle_intersect*vx_circle_intersect + vy_circle_intersect*vy_circle_intersect + vz_circle_intersect*vz_circle_intersect)
#
#     vx = vx_circle_intersect / vmag
#     vy = vy_circle_intersect / vmag
#     vz = vz_circle_intersect / vmag
#
#     foot_pt_x = (circle_intersect_2[0] + vx * (height_intersect_2_platform))
#     foot_pt_y = (circle_intersect_2[1] + vy * (height_intersect_2_platform))
#     foot_pt_z = (circle_intersect_2[2] + vz * (height_intersect_2_platform))
#
#     foot_pt = np.array([foot_pt_x, foot_pt_y, foot_pt_z])
#     print foot_pt
#
#     # ball = sphere(pos=(foot_pt), radius=5, material=materials.rough)
#     foot_1.pos = foot_pt
#
#     # scene.autocenter()
#     # display()
#     rate(30)
#     # time.sleep(2)
#
#     # allow the user to rotate and zoom the 3D image
#     # vPython_user_rotate()