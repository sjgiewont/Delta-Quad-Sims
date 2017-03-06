from blynkapi import Blynk
from time import sleep


auth_token = "340c28ef62d94a998855b7c8d4b89651"

blynk = Blynk(auth_token)
print blynk.app_status()


# create objects
height = Blynk(auth_token, pin = "V0")
x_pos = Blynk(auth_token, pin = "V1")
y_pos = Blynk(auth_token, pin = "V2")

# get current status
while(1):
    curr_height = height.get_val()
    curr_x_pos = x_pos.get_val()
    curr_y_pos = y_pos.get_val()

    print curr_height, curr_x_pos, curr_y_pos
    sleep(0.01)


