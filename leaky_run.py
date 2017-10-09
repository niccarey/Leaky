#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Libraries
from picamera import PiCamera
from picamera.array import PiRGBArray
import imutils
from imutils.video import VideoStream
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from pyfirmata import Arduino, util
from navImage import NavImage

import RPi.GPIO as gp
import random
import atexit
import time
import cv2
import os

import numpy as np
import threading 

from LeakyBot import LeakyBot


# initialisation: any global variables, etc

# Motor initialisation:
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)

# Camera initialization
webcam = VideoStream(src=0).start()
picam = VideoStream(usePiCamera=True, resolution=(640,480)).start()

time.sleep(0.1)
# Arduino intialisation
board = Arduino('/dev/ttyACM0')
time.sleep(1)
it = util.Iterator(board)
it.start()

# Filtering constants
erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)

l_red_left = np.array([170, 180, 0])
u_red_left = np.array([180, 255, 255])
l_red_right = np.array([0, 200, 0])
u_red_right = np.array([10, 255, 255])

l_blue_left = np.array([110, 0, 20])
u_blue_left = np.array([150, 100, 255])
l_blue_right = np.array([100, 0, 0])
u_blue_right = np.array([150, 100, 255])

# Matched filtering:
kernsize = 17
temp_size = 40
temp_blur_size = 11
corner_template = np.zeros((240,temp_size))
corner_template[30:240,1:20] = 1

left_template = np.array(255*cv2.GaussianBlur(corner_template, (kernsize, kernsize),0), dtype=np.uint8)
right_template = cv2.flip(left_template, 1)


# shuts down motors on program exit
def turnOffMotors():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)
    webcam.stop()
    picam.stop()
    cv2.destroyAllWindows()


atexit.register(turnOffMotors)

# Image filters

# wall balance and home balance could be combined into one function with 
# some alterations

def balance_detection(left_frame, right_frame, lb_left, lb_right, ub_left, ub_right, lscale, rscale, l_offset, r_offset):
    leftIm = NavImage(left_frame)
    rightIm = NavImage(right_frame)
    
    leftIm.convertHsv()
    rightIm.convertHsv()
    
    
    leftIm.hsvMask(lb_left, ub_left)
    rightIm.hsvMask(lb_right, ub_right)
    leftIm.erodeMask(erode_kernel, 1)
    rightIm.erodeMask(erode_kernel,1)
    rightIm.dilateMask(dilate_kernel, 1)
    
    left_weight = leftIm.maskWeight(lscale, l_offset)
    right_weight = rightIm.maskWeight(rscale, r_offset)
    
    weight_diff = left_weight - right_weight
    
    return (weight_diff, left_weight, right_weight)


def find_edge(nav_frame, l_bound, u_bound, thresh_state, direction, location, rcam_val=0):
    nav_image = NavImage(nav_frame)    

    nav_image.convertHsv()

    temp_debug_image = nav_image.frame
    nav_image.hsvMask(l_bound, u_bound)
    nav_image.erodeMask(erode_kernel, 1)
    
    if rcam_val:
        nav_image.dilateMask(dilate_kernel, 1)
        
    blur_image = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)
    if direction == 'left':
        template_match = cv2.matchTemplate(blur_image, left_template, cv2.TM_CCORR_NORMED)
        
    else:
        template_match = cv2.matchTemplate(blur_image, right_template, cv2.TM_CCORR_NORMED)
            
    # Locate best template match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    cv2.circle(blur_image, (max_loc[0]+temp_size//2 ,max_loc[1]), 7, (255,0,0), -1)    

    cv2.imshow("Navigation debug", blur_image)
    
    if (max_val > 0.8) and (max_loc[0]>10): # we are fairly sure we have found a good match
        # get true edge location
        # Could combine these but looks ugly
        
        if thresh_state == 'leq' and (max_loc[0] + temp_size//2 < location):
            print("Passed threshold. leq")
            return True

        elif thresh_state == 'geq' and (max_loc[0] + temp_size//2 > location):
            print("Passed threshold, geq")
            return True  
            
        else:
            return False
    
    else:
        return False
                
            
    

def main():
	# Initialise cameras - hold auto white balance constant to improve filtering
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    os.system('v4l2-ctl --set-ctrl=white_balance_temperature_auto=0')
    os.system('v4l2-ctl --set-ctrl=white_balance_temperature=2800')
    os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
    os.system('v4l2-ctl --set-ctrl=exposure_absolute=200')
    os.system('v4l2-ctl --set-ctrl=brightness=0')    

    
    sensing_pause = 1
    
    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:5:i')
    

    leaky1 = LeakyBot(myMotor1, myMotor2)
    #leaky1.get_graph().draw('state_diagram.png', prog='dot')
    leaky1.speed = 140
    leaky1.direction = 'fwd'
    
    # Set navigation thresholds:
    found_edge_threshold = 400
    back_up_threshold = 520
    leaky1.threshold_state = 'leq'
    
    start_turning_frame = 0
    sensor_temporary_loop = 0
    
    running = True
    
    # should be able to improve state handling
    # make sure set_motor_values is not called in a continuous loop UNLEss 
    # called with two arguments
    print("Waiting for block ...")
    
    while running:
        # should be a faster method than if/elif for handling state?
        
        # for debugging/testing: send camera images to screen (when relevant)        
        key = cv2.waitKey(1) & 0xFF        
        #print(leaky1.state)
            
        if leaky1.state in ['waiting', 'sensing']:
            if leaky1.state == 'waiting':
                try:
                    # read voltage
                    block_trigger = 1.0 - block_pin.read()
                    if block_trigger > 0.05:
                        # Pause to give hand time to get out of the way
                        time.sleep(2)
                        leaky1.have_block = True
                        leaky1.high_humidity = True
    
                        # pick a direction randomly
                        turn_dir = bool(random.getrandbits(1))
                        if turn_dir:
                            leaky1.direction = 'left'
                        else:
                            leaky1.direction = 'right'
                        
                        # make turning state safe 
                        # (there's got to be a better way of doing this!)
                        start_turning_frame = 0
                        print("Transition to turning")
                        leaky1.button_push()
                        print(leaky1.state)
                                                
                
                except:
                    print("problem reading board")
                    print(block_pin.read())        
                                        
                time.sleep(0.1)
    
    
            else:  
                # Don't bother using visuals to navigate, just limp forward                
                driving_time = 0
                sensing_time = 0
                leaky1.direction = 'fwd'
                leaky1.set_motor_values(leaky1.speed)
                
                while (driving_time < 0.5):                    
                    time.sleep(0.1)
                    driving_time = driving_time + 0.1
                    
                
                print("finished driving -> sensing")
                # now stop
                leaky1.set_motor_values(0)
                
                # some code here that reads our humidity sensors
                while (sensing_time < 4):
                    time.sleep(0.1)
                    # later: read_hum_sensors()
                    # if sensor_average < threshold
                    #   set high_humidity to false
                    #   trigger low_humidity
                    #   break while loop
                    sensing_time = sensing_time + 0.1
                
                print("finished sensing")
                
                # take this out once humidity code has been put in
                sensor_temporary_loop += 1
                # screw key press, let's just do it by time

                if sensor_temporary_loop > 3:
                    leaky1.high_humidity = False
                                        
                    # set direction: currently random (we may want to alternate left/right)
                    turn_dir = bool(random.getrandbits(1))

                    if turn_dir:
                        leaky1.direction = 'left'
                        print("depositing left")
                        leaky1.deposit_direction = 'left'
                        leaky1.threshold_state = 'geq'

                    else:
                        leaky1.direction = 'right'
                        print("depositing right")
                        leaky1.deposit_direction = 'right'
                        leaky1.threshold_state = 'leq'
                        
                    print("Transition to turning")
                    # pause briefly before each transition point
                    leaky1.set_motor_values(0)
                    time.sleep(2)

                    leaky1.low_humidity()
                     
    
        else:
            right_frame = webcam.read()
            left_frame = picam.read()
            #cv2.imshow("Left Camera", left_frame)            
            #cv2.imshow("Right Camera", right_frame)
                
            if leaky1.state == 'turning':
                # on_enter_turning handles motor directions and speeds
            
                if leaky1.high_humidity:              
                    # THREAD THIS
                    walls_match, left_wall, right_wall = balance_detection(left_frame, right_frame, l_blue_left, l_blue_right, u_blue_left, u_blue_right, 550000.0, 550000.0, 0.0, 5.0)
                    print("Balancing ...")
                    print(left_wall)
                    print(right_wall)

                    if start_turning_frame:
                        if (np.abs(walls_match) < 2) or (np.abs(np.sign(walls_match) + np.sign(wm_prev))< 0.5):

                            # Pause briefly before transition
                            leaky1.set_motor_values(0)
                            time.sleep(2)

                            # set trigger condition
                            print("Transition to sensing")
                            sensor_temporary_loop = 0
                            leaky1.walls_balanced()
                            print(leaky1.state)
                    
                    else:
                        start_turning_frame = 1
                    
                    # not ideal method, can probably improve
                    wm_prev = walls_match
                    
                elif leaky1.have_block:
                    # we are now turning to look for an edge

                    # THREAD THESE (maybe only take every x frame?)
                    if leaky1.direction == 'left':
                        turning_thresh = find_edge(left_frame, l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.direction, 640-found_edge_threshold, 0)
 
                    else:
                        turning_thresh = find_edge(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.direction, found_edge_threshold, 1)

                    if turning_thresh:
                        if leaky1.deposit_direction == 'left':
                            leaky1.threshold_state = 'leq'
                            
                        else:
                            leaky1.threshold_state = 'geq'

                        print("Transitioning to deposition")
                        
                        # Pause briefly before transition
                        leaky1.set_motor_values(0)
                        time.sleep(2)

                        leaky1.wall_found()
                
                else:
                    print("How did I get here? Entering wait state ..."
                    leaky1.close_to_home()                    
        
             
            elif leaky1.state == 'deposit':
                
                if leaky1.deposit_direction == 'left':
                    deposit_thresh = find_edge(left_frame, l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.deposit_direction, 640-back_up_threshold, 0)

                else:
                    deposit_thresh = find_edge(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.deposit_direction, back_up_threshold, 1)

                if deposit_thresh:
                    if leaky1.deposit_direction == 'left':
                        leaky1.threshold_state = 'geq'
                        
                    else:
                        leaky1.threshold_state = 'leq'

                    leaky1.have_block = False 
                    print("Backing up")
                    # pause briefly before each transition point
                    leaky1.set_motor_values(0)
                    time.sleep(2)

                    leaky1.reached_wall()
                
                
            elif leaky1.state == 'backup':
                # don't look for edge, look for home
                
                # Turning to look for red stripes
                home_match, left_home, right_home = balance_detection(left_frame, right_frame, l_red_left, l_red_right, u_red_left, u_red_right, 500000.0, 600000.0, 0.0, 1.5)
                    
                if np.abs(home_match) < 1.5:
                    # now entering go_home
                    print("Going home")
                    # pause briefly before each transition point
                    leaky1.set_motor_values(0)
                    time.sleep(2)

                    leaky1.home_spotted()


                
            elif leaky1.state == 'go_home':
                homing_var, left_home_val, right_home_val = balance_detection(left_frame, right_frame, l_red_left, l_red_right, u_red_left, u_red_right, 500000.0, 600000.0, 0.0, 1.5)
                
                if (left_home_val > 25) or (right_home_val > 25):
                    print("I'm home! Waiting ...")
                    leaky1.close_to_home()
                    
                else:
                    left_speed = leaky1.speed - 5*homing_var
                    right_speed = leaky1.speed + 5*homing_var
                    if np.abs(homing_var) > 3:
                        leaky1.set_motor_values(int(left_speed), int(right_speed))


            
        if key == ord("q"):
            break
    
    
    board.exit()
    webcam.stop()
    picam.stop()



if __name__ == '__main__':
	main()

