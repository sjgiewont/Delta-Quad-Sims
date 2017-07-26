'''
This is typically used on the BeagleBone Black to send serial commands to a servo controller. 

Requires the Adafruit BeagleBone Black IO library
'''

import numpy as np
import Adafruit_BBIO.UART as UART
import serial

# start a serial communication session between the BBB and the servo controller
def startSerial():
    UART.setup("UART1")
    ser = serial.Serial(port="/dev/ttyO1", baudrate=9600)
    ser.close()
    ser.open()
    if ser.isOpen():
        print "Serial is open!"
        return 0

# convert the angle the servo needs to be at to the servo value. Typical servo is mapped between a value of 500-2500
def angleToServoValue(thetas, leg_num):
    if leg_num == 1:
        servoValues = np.array([mapping(thetas[0],0,180,500,2500), mapping(thetas[1],0,180,500,2500), mapping(thetas[2],0,180,500,2500)])
    elif leg_num == 2:
        servoValues = np.array([mapping(thetas[0],0,180,500,2500), mapping(thetas[1],0,180,500,2500), mapping(thetas[2],0,180,500,2500)])
    elif leg_num == 3:
        servoValues = np.array([mapping(thetas[0],0,180,500,2500), mapping(thetas[1],0,180,500,2500), mapping(thetas[2],0,180,500,2500)])
    elif leg_num == 4:
        servoValues = np.array([mapping(thetas[0],0,180,500,2500), mapping(thetas[1],0,180,500,2500), mapping(thetas[2],0,180,500,2500)])
    else:
        return ValueError

    if servoValues[0] > 2400:
        servoValues[0] = 2400
    elif servoValues[0] < 600:
        servoValues[0] = 600

    if servoValues[1] > 2400:
        servoValues[1] = 2400
    elif servoValues[1] < 600:
        servoValues[1] = 600

    if servoValues[2] > 2400:
        servoValues[2] = 2400
    elif servoValues[2] < 600:
        servoValues[2] = 600

    return servoValues

# used to map one scale to another scale
def mapping(value, fromLow, fromHigh, toLow, toHigh):
    return (((value - fromLow) * (toHigh - toLow)) / (fromHigh - fromLow)) + toLow

# create a string consisting of all the servo values, mapped to a servo number
def serialSend(leg_1_thetas, leg_2_thetas, leg_3_thetas, leg_4_thetas):
    cmd1 = "#1 P%d #2 P%d #3 P%d " % (leg_1_thetas[0], leg_1_thetas[1], leg_1_thetas[2])
    cmd2 = "#4 P%d #5 P%d #6 P%d" % (leg_2_thetas[0], leg_2_thetas[1], leg_2_thetas[2])
    cmd3 = "#7 P%d #8 P%d #9 P%d" % (leg_3_thetas[0], leg_3_thetas[1], leg_3_thetas[2])
    cmd4 = "#10 P%d #11 P%d #12 P%d" % (leg_4_thetas[0], leg_4_thetas[1], leg_4_thetas[2])

    final_cmd = " ".join((cmd1, cmd2, cmd3, cmd4))
    print final_cmd



# test routine
leg1 = np.array([1111, 2222, 3333, 4444])
leg2 = np.array([1111, 2222, 3333, 4444])
leg3 = np.array([1111, 2222, 3333, 4444])
leg4 = np.array([1111, 2222, 3333, 4444])

serialSend(leg1, leg2, leg3, leg4)