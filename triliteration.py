from visual import *
import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm

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


# length from center of base to axis of rotation
base_dist = 55
# length of arm and leg
arm = 100
leg = 200

# define the position where the arms meet the base
base_p1 = np.array([55*cos(np.deg2rad(90)),55*sin(np.deg2rad(90)),0])
base_p2 = np.array([55*cos(np.deg2rad(210)),55*sin(np.deg2rad(210)),0])
base_p3 = np.array([55*cos(np.deg2rad(330)),55*sin(np.deg2rad(330)),0])

# draw the pivot points of the base
ball = sphere(pos=(base_p1), radius=10,  material=materials.rough)
ball = sphere(pos=(base_p2), radius=10, material=materials.rough)
ball = sphere(pos=(base_p3), radius=10, material=materials.rough)

# the angle each of the arms are oriented. 180 deg is horizontal
theta_1 = 180
theta_2 = 180
theta_3 = 180

# the first knee, in the universal y,z plane, pointint towards 12 o'clock
k1 = base_p1 + np.array([0, -arm*cos(np.deg2rad(theta_1)), arm*sin(np.deg2rad(theta_1))])

# second knee, pointing near 7 o'clock
k2 = base_p2 + np.array([0, -arm*cos(np.deg2rad(theta_2)), arm*sin(np.deg2rad(theta_2))])
k2 = np.array([k2[0]*cos(np.deg2rad(120)) - k2[1]*sin(np.deg2rad(120)) + base_p2[0] - cos(np.deg2rad(120))*base_p2[0] + sin(np.deg2rad(120))*base_p2[1], k2[0]*sin(np.deg2rad(120)) + k2[1]*cos(np.deg2rad(120)) + base_p2[1] - sin(np.deg2rad(120))*base_p2[0] - cos(np.deg2rad(120))*base_p2[1], k2[2]])

# third knee, pointing near 5 o'clock
k3 = base_p3 + np.array([0, -arm*cos(np.deg2rad(theta_3)), arm*sin(np.deg2rad(theta_3))])
k3 = np.array([k3[0]*cos(np.deg2rad(240)) - k3[1]*sin(np.deg2rad(240)) + base_p3[0] - cos(np.deg2rad(240))*base_p3[0] + sin(np.deg2rad(240))*base_p3[1], k3[0]*sin(np.deg2rad(240)) + k3[1]*cos(np.deg2rad(240)) + base_p3[1] - sin(np.deg2rad(240))*base_p3[0] - cos(np.deg2rad(240))*base_p3[1], k3[2]])

# draw lines to show arms
arm_link_1 = curve(pos=[(base_p1), (k1)])
arm_link_2 = curve(pos=[(base_p2), (k2)])
arm_link_3 = curve(pos=[(base_p3), (k3)])

# find the intersection of the three arm spheres
pointA, pointB = trilaterate(k1, k2, k3, leg, leg, leg)
print pointA, pointB

# find the center of the two possible intersection points
# center_point = (pointA + pointB) / 2
# print center_point

# draw lines to show arms
arm_link_1 = curve(pos=[(k1), (pointB)])
arm_link_2 = curve(pos=[(k2), (pointB)])
arm_link_3 = curve(pos=[(k3), (pointB)])


# allow the user to rotate and zoom the 3D image
vPython_user_rotate()