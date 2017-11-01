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
from sht_sensor import Sht as ST
import RPi.GPIO as gp

import random
import atexit
import time
import cv2
import os

import numpy as np
import threading 
import curses

from LeakyBot import LeakyBot


# initialisation: any global variables, etc 

# Motor initialisation:
print("Initialising motor hat ...")
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)
print("...done")


print("Initialising cameras ...")
# Camera initialization
webcam = VideoStream(src=0).start()
picam = VideoStream(usePiCamera=True, resolution=(640,480)).start()

time.sleep(0.1)
print("...done")

print("Initializing Arduino ...")
# Arduino intialisation
board = Arduino('/dev/ttyACM0')
time.sleep(0.5)
it = util.Iterator(board)
it.start()
print("... Arduino started")

print("Setting up filtering constants")
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
temp_size = 30
temp_blur_size = 11
corner_template = np.zeros((240,temp_size))
corner_template[:,1:15] = 1

left_template = np.array(255*cv2.GaussianBlur(corner_template, (kernsize, kernsize),0), dtype=np.uint8)
right_template = cv2.flip(left_template, 1)


# shuts down motors and cameras on program exit, cleans up terminal
def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)

    webcam.stop()
    picam.stop()
    cv2.destroyAllWindows()
    curses.endwin()


atexit.register(shutdownLeaky)

print("Establishing global variables")

# Image filters
deposit_edge = False
backup_edge = False

cX = 0
cY = 0

blue_walls_match = 50
home_match = 50
left_home = 0
right_home = 0
blue_left_wall = 0
blue_right_wall = 0

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


def find_edge(nav_frame, l_bound, u_bound, thresh_state, cam_flag, location):
    global cX, cY
    
    nav_image = NavImage(nav_frame) 
    nav_image.convertHsv()     
    nav_image.hsvMask(l_bound, u_bound)
    
    
    # additional operations: probably don't need
    nav_image.erodeMask(erode_kernel, 1) 
    nav_image.dilateMask(dilate_kernel, 1)
    blur_im = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)

    
    if cam_flag: #(direction == 'left'):
        template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), left_template, cv2.TM_CCORR_NORMED)
        
    else:
        template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), right_template, cv2.TM_CCORR_NORMED)
        #cv2.imshow("WTF", blur_im)
    
    # Locate best template match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    print max_val

    if (max_val > 0.7) and (max_loc[0]>2):
        # we are fairly sure we have found a good match
        # get true edge location
        cX = max_loc[0]+temp_size//2 
        cY = max_loc[1] 
        #print(cX)
        
        if thresh_state == 'leq' and (max_loc[0] + temp_size//2 < location):
            #print("Passed threshold. leq")
            return True
        
        elif thresh_state == 'geq' and (max_loc[0] + temp_size//2 > location):
            #print("Passed threshold, geq")
            return True  
            
        else:
            return False
    
    elif np.count_nonzero(blur_im) > 200000: # most of the screen is block
        print("uhoh, too close!")
        return True
        
    else:
        return False    

    
def lpfilter(input_buffer):
    # check length
    if len(input_buffer) < 3:
        return 0
    
    else:
        output = 0.6*input_buffer.pop() + 0.2*input_buffer.pop() + 0.2*input_buffer.pop()
        return output


# Should'nt need this now
def check_edges(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, found_edge_threshold, back_up_threshold):
    global deposit_edge
    global backup_edge
    
    deposit_edge = find_edge(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, found_edge_threshold)
    backup_edge = find_edge(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, back_up_threshold)
    if deposit_edge:
        print("deposit: TRUE")
        
    if backup_edge:
        print("Backup: TRUE")


def check_balance(lframe, rframe, ll_bound, ul_bound, lr_bound, ur_bound, left_scale, right_scale, left_offset, right_offset, wall_flag): #ubl, ubr, bl_scale, br_scale, lb_offset, rb_offset, lrl, lrr, url, urr, rl_scale, rr_scale, lr_offset, rr_offset):
    global blue_walls_match
    global home_match
    
    global left_home, right_home
    global blue_left_wall, blue_right_wall
    
    if wall_flag: 
        blue_walls_match, blue_left_wall, blue_right_wall = balance_detection(lframe, rframe, ll_bound, lr_bound, ul_bound, ur_bound, left_scale, right_scale, left_offset, right_offset)
        
    else: 
        home_match, left_home, right_home  = balance_detection(lframe, rframe, ll_bound, lr_bound, ul_bound, ur_bound, left_scale, right_scale, left_offset, right_offset)
    
    

def main(stdscr):
    
    stdscr.nodelay(1)
    curses.cbreak()
    curses.noecho()
    
    stdscr.addstr("Leaky started \n")
    
    global deposit_edge
    global backup_edge
    
	# Initialise cameras - hold auto white balance constant to improve filtering
    stdscr.addstr("Setting camera gains and white balance \n")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    os.system('v4l2-ctl --set-ctrl=white_balance_temperature_auto=0')
    os.system('v4l2-ctl --set-ctrl=white_balance_temperature=2800')
    os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
    os.system('v4l2-ctl --set-ctrl=exposure_absolute=200')
    os.system('v4l2-ctl --set-ctrl=brightness=0')    
    
    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:5:i')
    
    stdscr.addstr("Establishing humidity sensing GPIO settings \n")
    # Set up humidity sensing
    gp.setwarnings(False)
    gp.setmode(gp.BCM)

    gp.setup(11, gp.OUT)
    gp.setup(17, gp.IN)
    gp.setup(27, gp.IN)
    gp.setup(22, gp.IN)
    
    # usage: ST(clock pin, data pin) (default voltage)

    # have to use correct voltage option - default is 3.5V, otherwise
    # sens = ST(clock, data, voltage=ShtVDDLevel.vdd_5v) OR try
    # sens = ST(clock, data, '5V')

    sens1 = ST(11, 17)
    sens2 = ST(11, 27)
    sens3 = ST(11, 22)

    sens_array = [sens1, sens2, sens3]
    
    hum_threshold = 65
    
    # Set up motors
    stdscr.addstr("Setting motor parameters \n")
    leaky1 = LeakyBot(myMotor1, myMotor2)    
    leaky1.speed = 140
    leaky1.direction = 'fwd'
    
    stdscr.addstr("Setting navigation parameters \n")
    # Set navigation thresholds:
    found_edge_threshold = 350
    back_up_threshold = 550
    leaky1.threshold_state = 'leq'
    
    start_turning_frame = 0
    
    running = True
    
    fcount = 1
    # to-do: improve motor control setup
    
    winset = 0
    
    stdscr.addstr("Waiting for block ... \n")

    home_drive_flag = 0
    
    while running:
        # STATE CHECK
        #print("State check", leaky1.direction, leaky1.cam_flag, leaky1.threshold_state, leaky1.state)
        
        # if we have windows open, use waitkey
        if winset:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = stdscr.getch()
        
        # CHECK FOR BUTTON PUSHES
        if leaky1.is_waiting() or leaky1.is_deposit():
            try:
                block_trigger = 1.0 - block_pin.read()
                if block_trigger > 0.05:
                    stdscr.addstr(str(block_trigger))
                    stdscr.addstr(" \n")
                    leaky1.button_push()
                
            except Exception as e:
                stdscr.addstr("problem reading board, retry ... \n")
                print(e) # output to main terminal so we can see on exit
                            
        
        # FILTER CAMERA FRAMES
        right_frame = webcam.read()
        left_frame = picam.read()
        
        # If we are  - turning, backing up, depositing (ie doing anything that involves looking)
        # set wheel values
        # sleep for (0.1) or whatever
        # set wheel values to zero while we sense
        # while time < sense_time
        #    check wall balancing values
        #    check edge values
        #    trigger any requisite state changes
            

        #if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()) or (leaky1.is_go_home()):
            # figure out motor values
            # set motor values
        #    time.sleep(0.1)
        #    start_time = time.time()
            # set motor values to zero
        #    while (time.time() - start_time < sense_delay):
        #        if (leaky1.is_turning()):
                    
                # check blue wall balancing (don't thread)
                # OR
                # check red wall balancing
                # OR
                # do edge finding 
                
                # handle whatever has been trigered, depending on state
                # loop a few times in case the vision is lagging
                # now fall through

        # check for wall balancing (thread this so it interrupts)
        if (leaky1.is_turning()):
            leaky1.set_motor_values(0,0)
            balancetime = time.time()

            while (time.time() - balancetime) < 0.3 :
	        blue_thread = threading.Thread(target=check_balance, args=(left_frame, right_frame, l_blue_left, u_blue_left, l_blue_right, u_blue_right, 550000.0, 550000.0, 0.0, 5.0, 1, )).start()                

            leaky1.set_motor_values(leaky1.speed, leaky1.speed)
            time.sleep(0.1)
            
        elif (leaky1.is_backup() or leaky1.is_go_home()) and not home_drive_flag:
            balancetime =time.time()
            while (time.time() - balancetime) < 0.3 :
	        red_thread = threading.Thread(target=check_balance, args=(left_frame, right_frame, l_red_left, u_red_left, l_red_right, u_red_right, 500000.0, 600000.0, 0.0, 1.5, 0, )).start()

            home_drive_flag = 1


        if leaky1.is_turning():
            if blue_left_wall > 5: 
	        if (np.abs(blue_walls_match) < 2) or (np.abs(np.sign(blue_walls_match) + np.sign(blue_wm_prev))< 0.5):
        	    leaky1.walls_balanced()
                    # ensures each cycle starts with no edge found flag
                    deposit_edge = False
                    backup_edge = False

        if winset:
            cv2.imshow("Left camera", left_frame)
            cv2.imshow("Right camera", right_frame)
                                                
        blue_wm_prev = blue_walls_match
        
        if deposit_edge and leaky1.is_turning():
            leaky1.wall_found()
            backup_edge = False
            
        elif leaky1.start_turning_frame and (leaky1.is_turning() or leaky1.is_deposit()):
            # toggle turns on and off to let visual sensors catch up
            # manually hacking the background update cycle
            leaky1.set_motor_values(0,0)
            if leaky1.cam_flag:
                deposit_edge = find_edge(left_frame,  l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.cam_flag, 640-found_edge_threshold)
                backup_edge = find_edge(left_frame,  l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.cam_flag, 640-back_up_threshold)
                cv2.circle(left_frame, (cX,cY), 7, (255,0,0), -1)
        
            else:
                deposit_edge = find_edge(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.cam_flag, found_edge_threshold)
                backup_edge = find_edge(right_frame,  l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.cam_flag, back_up_threshold)
                cv2.circle(right_frame, (cX,cY), 7, (255,0,0), -1)
                
            if ccount > 5:
                leaky1.start_turning_frame = 0

            ccount += 1
        
        
        elif leaky1.is_turning() and not leaky1.high_humidity: 
            leaky1.start_turning_frame = 1
            ccount = 0                
            leaky1.on_enter_turning()
            time.sleep(0.1)
            
        elif leaky1.is_deposit():
            leaky1.set_motor_values(leaky1.speed, leaky1.speed)
            ccount += 1
            leaky1.start_turning_frame=1
            time.sleep(0.1)
            # quick hack
            if ccount > 5:
                backup_edge = True
        
        # hopefully that's all our turning code
        if backup_edge and leaky1.is_deposit():
            leaky1.reached_wall()
            deposit_edge = False
            backup_edge = False
        
        if (np.abs(home_match) < 1.5) and leaky1.is_backup() and (left_home > 2):
            leaky1.home_spotted()
            
        elif leaky1.is_go_home():            
            if (left_home > 25) or (right_home > 25):
                stdscr.addstr("I'm home! Waiting ... \n")
                leaky1.close_to_home()
                    
            elif home_drive_flag:
                left_speed = leaky1.speed - 2*home_match
                right_speed = leaky1.speed + 2*home_match
                if np.abs(home_match) > 3:
                    leaky1.set_motor_values(int(left_speed), int(right_speed))
                else:
                    leaky1.set_motor_values(leaky1.speed, leaky1.speed)

                time.sleep(0.1)
                home_drive_flag = 0

            else:
                leaky1.set_motor_values(0,0)

    
        # CHECK HUMIDITY DATA
        # nB: reading from the humidity sensors is SLOW
        
        if leaky1.is_sensing():
            if (time.time() - leaky1.sensing_clock < 10):
                # Read all sensors - ok here's a question
                # do I have to read the sensors for them to settle?
                
                # TODO: once we are sure humidity bubble is working
                # try just reading once, at the end of the sensing pause
                time.sleep(0.1)

                hum_count = 0                
                for sens_i in sens_array:
                    try:
                        temp_new = sens_i.read_t()
                        hum_new = sens_i.read_rh()
                        #print("Sensor read: ", hum_new)
                        #stdscr.addstr(" \n")
                        hum_count +=1
                        
                    except Exception as e:
                        print("Sensor problem: ", (hum_count+1))
                        #stdscr.addstr("Sensor problem \n")

                
            else: # last sensor reading
                stdscr.addstr("Entering final sensor read ...\n")
                hum_sum = 0
                hum_count = 0
                for sens_i in sens_array:
                    try:
                        temp_new = sens_i.read_t()
                        hum_new = sens_i.read_rh()
                        print("Final read: ", int(temp_new), int(hum_new))
                        hum_sum = hum_sum + hum_new
                        hum_count += 1
                        
                    except Exception as e:
                        print("Sensor problem: ", (hum_count+1))
                        stdscr.addstr("continue ... \n")

                        #print(e) # output to main terminal to catch on exit
                    
                if hum_count > 0:
                    hum_av = hum_sum/hum_count
                    print("Average humidity: ", int(hum_av))
                    stdscr.addstr(" \n")
                    
                    if hum_av < hum_threshold:
                        leaky1.high_humidity = False
                        leaky1.low_humidity()
                                
                    else:
                        # transitions to driving
                        leaky1.humidity_maintained()
                
                else:
                    stdscr.addstr("no sensors available, starting again \n")
                    leaky1.sensing_clock = time.time()

                

        elif leaky1.is_driving():
            if (time.time() - leaky1.driving_clock < 0.5):
                time.sleep(0.1)
            
            else:
                leaky1.stop_driving()
        
               
        fcount += 1
        
        if key == ord("q"):
            running=False
            break
    
    
    board.exit()
    webcam.stop()
    picam.stop()
    shutdownLeaky()



if __name__ == '__main__':
	curses.wrapper(main)

