'''
This calculates the inverse kinematics of a classic Delta robot, NOT a Delta-Quad
'''

import numpy as np
from math import sqrt,pow,degrees,atan,pi, tan, cos, sin

def inverseKinematic(x, y, z, arm, leg):
    """Calculate the inverse kinematic of the robot for a position x0, y0, z0 and returns theta1, theta2 and theta3"""
    #global theta1, theta2, theta3

    theta1 = angle_xy(x, y, z, arm, leg)
    x2 = (x * -0.5) + (y * (sqrt(3) / 2))
    y2 = (y * -0.5) - (x * (sqrt(3) / 2))
    theta2 = angle_xy(x2, y2, z, arm, leg)
    x3 = (x * -0.5) - (y * (sqrt(3) / 2))
    y3 = (y * -0.5) + (x * (sqrt(3) / 2))
    theta3 = angle_xy(x3, y3, z, arm, leg)
    return theta1, theta2, theta3


def angle_xy(x0, y0, z0, arm, leg):
    e = 6.5
    f = 16.5
    re = arm
    rf = leg
    # re = 52.4       #leg length
    # rf = 17.145     #arm length

    y1 = -0.5 * 0.57735 * f
    y0 = y0 - (0.5 * 0.57735 * e)
    anp = ((x0 * x0) + (y0 * y0) + (z0 * z0) + (rf * rf))
    ann = ((re * re) + (y1 * y1))
    an = anp - ann
    ad = 2 * z0
    a = an / ad
    b = (y1 - y0) / z0
    d = -(a + b * y1) * (a + b * y1) + rf * (b * b * rf + rf)
    """print("d={0} y1={1} y0={2} a={3} b={4} an={5} ad={6} anp={7} ann={8}".format(d,y1,y0,a,b,an,ad,anp,ann))"""

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


def maxWorkspace(arm, leg):
    theta_list_1 = []
    theta_list_2 = []
    theta_list_3 = []

    for x in range(0,60):
        for y in range(0, 60):
            for z in range(1, 60, 1):
                theta1, theta2, theta3 = inverseKinematic(x, y, z, arm, leg)
                if theta1 is not None and theta2 is not None and theta3 is not None:
                    theta_list_1.append(x)
                    theta_list_2.append(y)
                    theta_list_3.append(z)

    # print max(theta_list_1)
    # print max(theta_list_2)
    # print max(theta_list_3)

    return max(theta_list_1), max(theta_list_2), max(theta_list_3)


def forward_Kinematics(theta1, theta2, theta3, arm, leg):
    e = 6.5
    f = 16.5
    re = arm
    rf = leg

    sqrt3 = sqrt(3.0)
    tan60 = sqrt3
    sin30 = 0.5
    tan30 = 1 / sqrt3

    t = (f - e) * tan30 / 2
    dtr = pi /180.0

    theta1 *= dtr
    theta2 *= dtr
    theta3 *= dtr

    y1 = -(t + rf * cos(theta1))
    z1 = -rf * sin(theta1)

    y2 = (t + rf * cos(theta2)) * sin30
    x2 = y2 * tan(60)
    z2 = -rf * sin(theta2)

    y3 = (t + rf * cos(theta3)) * sin30
    x3 = -y3 * tan60
    z3 = -rf * sin(theta3)

    dnm = (y2 - y1) * x3 - (y3 - y1) * x2

    w1 = y1 * y1 + z1 * z1

    w2 = x2 * x2 + y2 * y2 + z2 * z2

    w3 = x3 * x3 + y3 * y3 + z3 * z3

    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0

    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0

    a = a1 * a1 + a2 * a2 + dnm * dnm
    b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
    c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + b1 * b1 + dnm * dnm * (z1 * z1 - re * re)

    #discriminant
    d = (b * b) - (4.0 * a * c)
    if (d < 0):
        return None

    z0 = -0.5 * (b + sqrt(d)) / a
    x0 = (a1 * z0 + b1) / dnm
    y0 = (a2 * z0 + b2) / dnm
    return x0, y0, z0


# Forward kinematics: (theta1, theta2, theta3) -> (x0, y0, z0)
#   Returned {error code,theta1,theta2,theta3}
def forward(theta1, theta2, theta3, arm, leg):
    # Trigonometric constants
    s = 165 * 2
    sqrt3 = sqrt(3.0)
    pi = 3.141592653
    sin120 = sqrt3 / 2.0
    cos120 = -0.5
    tan60 = sqrt3
    sin30 = 0.5
    tan30 = 1.0 / sqrt3

    e = 6.5
    f = 16.5
    re = leg
    rf = arm

    # e = 26.0
    # f = 69.0
    # re = 128.0
    # rf = 88.0

    x0 = 0.0
    y0 = 0.0
    z0 = 0.0

    t = (f - e) * tan30 / 2.0
    dtr = pi / 180.0

    theta1 *= dtr
    theta2 *= dtr
    theta3 *= dtr

    y1 = -(t + rf * cos(theta1))
    z1 = -rf * sin(theta1)

    y2 = (t + rf * cos(theta2)) * sin30
    x2 = y2 * tan60
    z2 = -rf * sin(theta2)

    y3 = (t + rf * cos(theta3)) * sin30
    x3 = -y3 * tan60
    z3 = -rf * sin(theta3)

    dnm = (y2 - y1) * x3 - (y3 - y1) * x2

    w1 = y1 * y1 + z1 * z1
    w2 = x2 * x2 + y2 * y2 + z2 * z2
    w3 = x3 * x3 + y3 * y3 + z3 * z3

    # x = (a1*z + b1)/dnm
    a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
    b1 = -((w2 - w1) * (y3 - y1) - (w3 - w1) * (y2 - y1)) / 2.0

    # y = (a2*z + b2)/dnm
    a2 = -(z2 - z1) * x3 + (z3 - z1) * x2
    b2 = ((w2 - w1) * x3 - (w3 - w1) * x2) / 2.0

    # a*z^2 + b*z + c = 0
    a = a1 * a1 + a2 * a2 + dnm * dnm
    b = 2.0 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm * dnm)
    c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + b1 * b1 + dnm * dnm * (z1 * z1 - re * re)

    # discriminant
    d = b * b - 4.0 * a * c
    if d < 0.0:
        return [1, 0, 0, 0]  # non-existing povar. return error,x,y,z

    z0 = -0.5 * (b + sqrt(d)) / a
    x0 = (a1 * z0 + b1) / dnm
    y0 = (a2 * z0 + b2) / dnm

    return [0, x0, y0, z0]
