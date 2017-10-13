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
deposit_edge = False
backup_edge = False

deposit_edge_buffer = []
backup_edge_buffer = []

blue_walls_match = 50
home_match = 50
left_home = 0
right_home = 0


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
    #cv2.circle(blur_image, (max_loc[0]+temp_size//2 ,max_loc[1]), 7, (255,0,0), -1)    
    
    if (max_val > 0.8) and (max_loc[0]>10): # we are fairly sure we have found a good match
        # get true edge location
        # Could combine these but looks ugly
        # should actually return the value, then lowpass filter in the calling function
        return max_loc[0]+temp_size//2
            
    else:
        return 0
    
    
def lpfilter(input_buffer):
    # check length
    if len(input_buffer) < 3:
        return 0
    
    else:
        output = 0.6*input_buffer.pop() + 0.2*input_buffer.pop() + 0.2*input_buffer.pop()
        return output


def check_edges(frame, l_blue, u_blue, threshold_state, direction , deposit_direction, found_edge_threshold, back_up_threshold, dir_flag):
    global deposit_edge
    global backup_edge
    global deposit_edge_buffer, backup_edge_buffer
    
    # some global buffer containing at least three values
    deposit_location = find_edge(frame, l_blue, u_blue, threshold_state, direction, found_edge_threshold, dir_flag)
    backup_location = find_edge(frame, l_blue, u_blue, threshold_state, deposit_direction, back_up_threshold, dir_flag)
   
    if deposit_location is not 0:
        if threshold_state == 'leq' and (deposit_location < location):
            # push value into buffer
            deposit_edge_buffer.append(deposit_location)

        elif thresh_state == 'geq' and (deposit_location > location):
            # push value into buffer
            deposit_edge_buffer.append(deposit_location)

    if backup_location is not 0:
        if threshold_state == 'leq' and (backup_location < location):
            # push value into buffer
            backup_edge_buffer.append(backup_location)

        elif thresh_state == 'geq' and (backup_location > location):
            # push value into buffer
            backup_edge_buffer.append(backup_location)
            
    filtered_deposit = lpfilter(deposit_edge_buffer)
    filtered_backup = lpfilter(backup_edge_buffer)
    
    if filtered_deposit is not 0:
        deposit_edge = True
    else:
        deposit_edge = False
        
    if filtered_backup is not 0:
        deposit_edge = True
    else:
        deposit_edge = False



def check_balance(lframe, rframe, lbl, lbr, ubl, ubr, bl_scale, br_scale, lb_offset, rb_offset, lrl, lrr, url, urr, rl_scale, rr_scale, lr_offset, rr_offset):
    global blue_walls_match
    global home_match
    
    global left_home, right_home

    blue_walls_match, blue_left_wall, blue_right_wall = balance_detection(lframe, rframe, lbl, lbr, ubl, ubr, bl_scale, br_scale, lb_offset, rb_offset)
    home_match, left_home, right_home = balance_detection(lframe, rframe, lrl, lrr, url, urr, rl_scale, rr_scale, lr_offset, rr_offset)
    

    

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

    
    # low-pass filter the visual feedback
    
    
    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:5:i')
    

    leaky1 = LeakyBot(myMotor1, myMotor2)    
    leaky1.speed = 140
    leaky1.direction = 'fwd'
    
    # Set navigation thresholds:
    found_edge_threshold = 400
    back_up_threshold = 520
    leaky1.threshold_state = 'leq'
    
    start_turning_frame = 0
    sensor_temporary_loop = 0
    
    running = True
    
    # to-do: improve motor control setup
    
    print("Waiting for block ...")
    
    while running:
        # poll each event, all functional transition code should be in the class
        
        key = cv2.waitKey(1) & 0xFF
        
        # CHECK FOR BUTTON PUSHES
        # have to hold this in a loop because overall loop too slow??
        if leaky1.is_waiting() or leaky1.is_deposit():
            try:
                block_trigger = 1.0 - block_pin.read()
                if block_trigger > 0.05:
                    print(block_trigger)
                    leaky1.button_push()
                
            except:
                print("problem reading board, retry ...")
            
        
        # FILTER CAMERA FRAMES
        right_frame = webcam.read()
        left_frame = picam.read()
        
        cv2.imshow("Left camera", left_frame)
        cv2.imshow("Right camera", right_frame)
        
        # check for wall balancing
        bal_thread = threading.Thread(target=check_balance, args=(left_frame, right_frame, l_blue_left, l_blue_right, u_blue_left, u_blue_right, 550000.0, 550000.0, 0.0, 5.0, l_red_left, l_red_right, u_red_left, u_red_right, 500000.0, 600000.0, 0.0, 1.5,))
        bal_thread.start()
        # look for edge data
        # possibly: want to transition left vs right handling to state machine?
        
        if leaky1.direction == 'left':
            edge_thread = threading.Thread(target=check_edges, args=(left_frame, l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.direction ,leaky1.deposit_direction, 640-found_edge_threshold, 640-back_up_threshold, 0,))
            edge_thread.start()
 
        else:
            edge_thread = threading.Thread(target=check_edges, args=(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.direction ,leaky1.deposit_direction, found_edge_threshold, back_up_threshold, 1,))
            edge_thread.start()

        if leaky1.start_turning_frame and leaky1.is_turning():
            # toggle turns on and off to let visual sensors catch up
            # bit of a hack
            leaky1.set_motor_values(0,0)
            leaky1.start_turning_frame = 0
            if (np.abs(blue_walls_match) < 2) or (np.abs(np.sign(blue_walls_match) + np.sign(blue_wm_prev))< 0.5):
                    leaky1.walls_balanced()

        elif leaky1.is_turning():
            leaky1.start_turning_frame = 1
            leaky1.on_enter_turning()
            time.sleep(0.2)
                
        blue_wm_prev = blue_walls_match
        
        if deposit_edge and leaky1.is_turning():
            leaky1.wall_found()
        
        
        
        if backup_edge and leaky1.is_deposit():
            leaky1.reached_wall()


        if (np.abs(home_match) < 1.5) and leaky1.is_backup():
            leaky1.home_spotted()
            
        elif leaky1.is_go_home():
            if (left_home > 25) or (right_home > 25):
                print("I'm home! Waiting ...")
                leaky1.close_to_home()
                    
            else:
                left_speed = leaky1.speed - 5*home_match
                right_speed = leaky1.speed + 5*home_match
                if np.abs(home_match) > 3:
                    leaky1.set_motor_values(int(left_speed), int(right_speed))
    
        # CHECK HUMIDITY DATA
        
        # Reading from the humidity sensors can be slow, so we probably
        # don't want to do it unless we're in the appropriate state. 
        # slightly hacky but will work for now
        
        if leaky1.is_sensing():
            if (time.time() - leaky1.sensing_clock < 4):
                time.sleep(0.1)
                # later: read_hum_sensors()
                # if sensor_average < threshold
                #   set high_humidity to false
                #   trigger low_humidity
                #   break while loop
            
            else:
                if leaky1.sensor_loop < leaky1.sensor_loop_max:
                    # transitions to driving
                    leaky1.sensor_loop += 1
                    leaky1.humidity_maintained()
                
                else:
                    # probably a better location to put this, will figure out when
                    # have humidity sensors                
                    leaky1.high_humidity = False
                    leaky1.low_humidity()
                     

        elif leaky1.is_driving():
            if (time.time() - leaky1.driving_clock < 0.5):
                time.sleep(0.1)
            
            else:
                leaky1.stop_driving()
        
               
            
        if key == ord("q"):
            break
    
    
    board.exit()
    webcam.stop()
    picam.stop()



if __name__ == '__main__':
	main()

