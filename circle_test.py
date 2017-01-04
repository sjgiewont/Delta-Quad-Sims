import numpy as np
import pylab #Imports matplotlib and a host of other useful modules

foot_radius = 25  # 25
leg = 200

circle_1 = np.array([-50, -75, 200])
circle_2 = np.array([200, 0, 200])

d = np.hypot(circle_2[0]-circle_1[0], circle_2[1]-circle_1[1])

x = (circle_1[2]*circle_1[2] - circle_2[2]*circle_2[2] + d*d)/(2*d)
y = np.sqrt(circle_1[2]*circle_1[2] - x*x)

ex = (circle_2[0] - circle_1[0]) / d
ey = (circle_2[1] - circle_1[1]) / d

P1 = np.array([circle_1[0] + x * ex - y * ey, circle_1[1] + x * ey + y * ex])
P2 = np.array([circle_1[0] + x * ex + y * ey, circle_1[1] + x * ey - y * ex])

print P1, P2

theta_start_1 = np.degrees(np.arctan2(P2[1]-circle_1[1],P2[0] - circle_1[0]))
theta_start_2 = np.degrees(np.arctan2(P2[1] - circle_2[1],P2[0] - circle_2[0]))
print "theta_start", theta_start_1, theta_start_2

circle_angle_1 = np.degrees(np.arctan2(circle_2[1]-circle_1[1],circle_2[0] - circle_1[0]))
circle_angle_2 = np.degrees(np.arctan2(circle_1[1]-circle_2[1],circle_1[0] - circle_2[0]))

print "circle angle", circle_angle_1, circle_angle_2

sin_theta = np.sqrt((leg*leg) - (0.25 * (d - foot_radius) * (d - foot_radius))) / leg
trap_theta = np.degrees(np.arcsin(sin_theta))
print "trap angle", (trap_theta)

theta_mod_1 = circle_angle_1 - trap_theta
theta_mod_2 = circle_angle_2 + trap_theta
print "theta mod", theta_mod_1, theta_mod_2


if theta_start_1 > -90:
    f1 = np.array([circle_1[0] + leg*np.cos(np.deg2rad(theta_mod_1)), circle_1[1] + leg*np.sin(np.deg2rad(theta_mod_1))])
else:
    f1 = np.array([circle_1[0] + leg*np.cos(np.deg2rad(theta_mod_1)), circle_1[1] + leg*np.sin(np.deg2rad(theta_mod_1))])

if theta_start_2 > -90:
    f2 = np.array([circle_2[0] + leg*np.cos(np.deg2rad(theta_mod_2)), circle_2[1] + leg*np.sin(np.deg2rad(theta_mod_2))])
else:
    f2 = np.array([circle_2[0] + leg*np.cos(np.deg2rad(theta_mod_2)), circle_2[1] + leg*np.sin(np.deg2rad(theta_mod_2))])


platform_length = np.hypot((f1[0]-f2[0]), (f1[1]-f2[1]))
print platform_length

print f1, f2

# find slope of platform line
platform_a = f1[1] - f2[1]
platform_b = f2[0] - f1[0]
platform_c = (f1[0] - f2[0]) * f1[1] + (f2[1] - f1[1]) * f1[0]

print platform_a, platform_b, platform_c

circle_2_platform_dist = (np.abs(platform_a*P2[0] + platform_b*P2[1] + platform_c)) / np.sqrt(platform_a*platform_a + platform_b*platform_b)
print circle_2_platform_dist

height_intersect = np.sqrt(leg*leg - 0.25*(d*d))
height_platform = np.sqrt(leg*leg - 0.25*(d - foot_radius)*(d - foot_radius))
intersect_2_platform_dist = height_platform - height_intersect
print intersect_2_platform_dist


# find point along a line, given a distance. Uses vectors
vx_circle_intersect = P2[0] - P1[0]
vy_circle_intersect = P2[1] - P1[1]

vmag = np.sqrt(vx_circle_intersect*vx_circle_intersect + vy_circle_intersect*vy_circle_intersect)

vx = vx_circle_intersect / vmag
vy = vy_circle_intersect / vmag

foot_pt_x = (P2[0] + vx * (intersect_2_platform_dist))
foot_pt_y = (P2[1] + vy * (intersect_2_platform_dist))
foot_pt = np.array([foot_pt_x, foot_pt_y])
print foot_pt




# plot visuals

# line = pylab.plot(f1, f2)
# point = pylab.plot(P1[0], P1[1])

cir1 = pylab.Circle((circle_1[0],circle_1[1]), radius=circle_1[2],  fc='y') #Creates a patch that looks like a circle (fc= face color)
cir2 = pylab.Circle((circle_2[0],circle_2[1]), radius=circle_2[2], alpha =.2, fc='b') #Repeat (alpha=.2 means make it very translucent)

ax = pylab.axes(aspect=1) #Creates empty axes (aspect=1 means scale things so that circles look like circles)
pylab.xlim(-400, 400)
pylab.ylim(-400, 400)

l = pylab.Line2D([f1[0],f2[0]],[f1[1], f2[1]])
ax.add_line(l)

l2 = pylab.Line2D([P2[0], foot_pt[0]], [P2[1], foot_pt[1]])
ax.add_line(l2)

ax.add_patch(cir1) #Grab the current axes, add the patch to it
ax.add_patch(cir2) #Repeat

pylab.show()