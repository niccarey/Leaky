from Adafruit_MotorHAT import Adafruit_MotorHAT
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import threading 

import atexit
import time

from LeakyBot import LeakyBot

mh = Adafruit_MotorHAT(addr=0x60)

myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)


# Preliminary functions
def turnOffMotors():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)


def main():	
    # set some speed
    des_speed = 140
    
    fcount = 1
    leaky1 = LeakyBot(myMotor1, myMotor2)
    leaky1.speed = 180
    leaky1.direction = 'fwd'
    running = True

    while running: 

        driving_time = 0
        sensing_time = 0
        leaky1.direction = 'fwd'
        leaky1.set_motor_values(leaky1.speed, leaky1.speed)
                
        while (driving_time < 1):                    
            time.sleep(0.1)
            driving_time = driving_time + 0.1
                    
                
        print("finished driving -> sensing")
        # now stop
        leaky1.set_motor_values(0,0)
                
        # some code here that reads our humidity sensors
        while (sensing_time < 4):
            time.sleep(0.1)
            sensing_time = sensing_time + 0.1
                
        print("finished sensing")

        fcount += 1
        
        if fcount > 6:
            running = False

if __name__ == "__main__":
	main()
	
