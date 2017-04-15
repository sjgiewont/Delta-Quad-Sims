import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi

def piecewiseMotion_2(step_length, step_height, step_angle, leg_height, step_precision):
    # step_length = the distance the leg will step relative to the origin
    # step_height = the distance the leg will step above the height of the leg (leg_height)
    # step_angle = the angle about the origin the stepping motion will take place
    # leg_height = the height the leg will be dragging at (should be a negative number)
    # step_precision = the number of step increments the motion will be broken into (the more the better precision)

    # define the percentage of time it takes to lift the leg
    step_up_time = float(0.2)

    # define the two end points based on the step length and the angle of the step
    pos_0 = np.array([-step_length*np.sin(np.deg2rad(step_angle)), -step_length*np.cos(np.deg2rad(step_angle)), leg_height])
    pos_1 = np.array([step_length*np.sin(np.deg2rad(step_angle)), step_length*np.cos(np.deg2rad(step_angle)), leg_height])

    # create increments of time
    t = np.linspace(0, 1, step_precision)

    # the rate of change in relation to the step up timeto the step up time
    m_drag = float(1) / (1 - step_up_time)
    b_drag = -(m_drag * step_up_time)

    m_step = float(1) / step_up_time

    # the rate of change and starting point of the dragging motion wrt
    # define a matrix of Z axis parabolic conditions and its arguments, need to solve for constants of this relation
    # z(t) = at^2 + bt + c  --> need to solve for a, b, c
    z_matrix = np.matrix([[0, 0, 1], [step_up_time**2, step_up_time, 1], [(step_up_time/2)**2, (step_up_time/2), 1]])
    z_conditions = np.matrix([[pos_0[2]], [pos_1[2]], [leg_height + step_height]])
    z_constants = z_matrix.I * z_conditions

    # determine all the values defined by the piecewise functions
    piecewise_x = np.piecewise(t, [(t >= 0) & (t <= step_up_time), t > step_up_time], [lambda t: (1 - m_step*t)*pos_0[0] + m_step*t*pos_1[0], lambda t: (1 - (m_drag*t + b_drag))*pos_1[0] + (m_drag*t + b_drag)*pos_0[0]])
    piecewise_y = np.piecewise(t, [(t >= 0) & (t <= step_up_time), t > step_up_time], [lambda t: (1 - m_step*t)*pos_0[1] + m_step*t*pos_1[1], lambda t: (1 - (m_drag*t + b_drag))*pos_1[1] + (m_drag*t + b_drag)*pos_0[1]])
    piecewise_z = np.piecewise(t, [(t >= 0) & (t <= step_up_time), (t > step_up_time)], [lambda t: z_constants[0,0]*t**2 + z_constants[1,0]*t + z_constants[2,0], lambda t: leg_height])

    plt.plot(t, piecewise_x)
    plt.show()
    plt.plot(t, piecewise_y)
    plt.show()
    plt.plot(t, piecewise_z)
    plt.show()

    piecewise = []

    # create matrix of all positions along trajectory
    for i in range(len(t)):
        piecewise.append([piecewise_x[i], piecewise_y[i], piecewise_z[i]])

    return piecewise

def piecewiseMotion_3(step_length, step_height, step_angle, leg_height, step_precision):
    # step_length = the distance the leg will step relative to the origin
    # step_height = the distance the leg will step above the height of the leg (leg_height)
    # step_angle = the angle about the origin the stepping motion will take place
    # leg_height = the height the leg will be dragging at (should be a negative number)
    # step_precision = the number of step increments the motion will be broken into (the more the better precision)

    # define the percentage of time it takes to lift the leg
    step_up_time = float(0.5)

    # define the two end points based on the step length and the angle of the step
    pos_0 = np.array([-step_length*np.sin(np.deg2rad(step_angle)), -step_length*np.cos(np.deg2rad(step_angle)), leg_height])
    pos_1 = np.array([step_length*np.sin(np.deg2rad(step_angle)), step_length*np.cos(np.deg2rad(step_angle)), leg_height])

    # create increments of time
    t = np.linspace(0, 1, step_precision)

    # the rate of change in relation to the step up timeto the step up time
    m_drag = float(1) / (1 - step_up_time)
    b_drag = -(m_drag * step_up_time)

    m_step = float(1) / step_up_time

    # the rate of change and starting point of the dragging motion wrt
    # define a matrix of Z axis parabolic conditions and its arguments, need to solve for constants of this relation
    # z(t) = at^2 + bt + c  --> need to solve for a, b, c
    z_matrix = np.matrix([[0, 0, 1], [step_up_time**2, step_up_time, 1], [(step_up_time/2)**2, (step_up_time/2), 1]])
    z_conditions = np.matrix([[pos_1[2]], [pos_0[2]], [leg_height + 5]])
    z_constants = z_matrix.I * z_conditions

    # determine all the values defined by the piecewise functions
    piecewise_x = np.piecewise(t, [(t >= 0) & (t <= step_up_time), t > step_up_time], [lambda t: (1 - m_step*t)*pos_0[0] + m_step*t*pos_1[0], lambda t: (1 - (m_drag*t + b_drag))*pos_1[0] + (m_drag*t + b_drag)*pos_0[0]])
    piecewise_y = np.piecewise(t, [(t >= 0) & (t <= step_up_time), t > step_up_time], [lambda t: (1 - m_step*t)*pos_0[1] + m_step*t*pos_1[1], lambda t: (1 - (m_drag*t + b_drag))*pos_1[1] + (m_drag*t + b_drag)*pos_0[1]])
    piecewise_z = np.piecewise(t, [(t >= 0) & (t <= step_up_time), (t > step_up_time)], [lambda t: leg_height + ((step_length / 2) * np.sin(pi - (2*(pi * t)))), lambda t: z_constants[0,0]*(0.5-t)**2 + z_constants[1,0]*(0.5-t) + z_constants[2,0]])

    plt.plot(t, piecewise_x)
    plt.show()
    plt.plot(t, piecewise_y)
    plt.show()
    plt.plot(t, piecewise_z)
    plt.show()

    piecewise = []

    # create matrix of all positions along trajectory
    for i in range(len(t)):
        piecewise.append([piecewise_x[i], piecewise_y[i], piecewise_z[i]])

    return piecewise

def piecewiseMotion(degrees, precision):
    # set the step size of the time parameter
    t = np.linspace(0, 1, precision)

    # convert the degree input to radians
    rad = degrees * (pi/180)

    # determine all the values defined by the piecewise functions
    piecewise_y = np.piecewise(t, [(t >= 0) & (t <= 0.2), t > 0.2], [lambda t: -np.cos(rad)+10*t*np.cos(rad), lambda t: 1.5*np.cos(rad)-2.5*t*np.cos(rad)])
    piecewise_x = np.piecewise(t, [(t >= 0) & (t <= 0.2), t > 0.2], [lambda t: -np.sin(rad)+10*t*np.sin(rad), lambda t: 1.5*np.sin(rad)-2.5*t*np.sin(rad)])
    piecewise_z = np.piecewise(t, [(t >= 0) & (t <= 0.1), (t > 0.1) & (t <= 0.2)], [lambda t: 10*t, lambda t: 2-10*t])

    piecewise = []

    # create matrix of all positions along trajectory
    for i in range(len(t)):
        piecewise.append([piecewise_x[i], piecewise_y[i], piecewise_z[i]])


    #
    # Plot the Piecewise Functions
    plt.plot(t, piecewise_x)
    plt.show()
    plt.plot(t, piecewise_y)
    plt.show()
    plt.plot(t, piecewise_z)
    plt.show()

    return piecewise