'''-----------------------------------------
This is the original script to test the ability to get data from the Blynk controller in real time.

Please install Blynk API first. https://github.com/xandr2/blynkapi

Will get values from the "virtual pins" made on the Blynk controller
-----------------------------------------'''

from blynkapi import Blynk
from time import sleep

auth_token = "340c28ef62d94a998855b7c8d4b89651"

# create the blynk object
blynk = Blynk(auth_token)

# print the status to ensure it is connected to Blynk server
print blynk.app_status()

# create virtual pin objects
height = Blynk(auth_token, pin = "V0")
x_pos = Blynk(auth_token, pin = "V1")
y_pos = Blynk(auth_token, pin = "V2")

# get current status of each pin
while(1):
    curr_height = height.get_val()
    curr_x_pos = x_pos.get_val()
    curr_y_pos = y_pos.get_val()

    print curr_height, curr_x_pos, curr_y_pos
    sleep(0.01)


