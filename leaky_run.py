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
from leaky_nav_functions import balance_detection, find_centroid

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

l_blue_left = np.array([100, 0, 20])
u_blue_left = np.array([130, 100, 255])
l_blue_right = np.array([100, 0, 0])
u_blue_right = np.array([150, 100, 255])

l_green_left = np.array([40, 100, 0])
u_green_left = np.array([70, 255, 255])
l_green_right = np.array([50, 100, 0])
u_green_right = np.array([90, 255, 255])


# Matched filtering:
kernsize = 17
temp_size = 30
temp_blur_size = 11
corner_template = np.zeros((120,temp_size))
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

# Navigation constants
blue_walls_match = 50
home_match = 50
left_home = 0
right_home = 0
blue_left_wall = 0
blue_right_wall = 0

    
# Set navigation thresholds:
found_edge_right = 500
back_up_right = 560
found_edge_left = 300
back_up_left = 230


def main():

#    stdscr.nodelay(1)
#    curses.cbreak()
#    curses.noecho()
    
#    stdscr.addstr("Leaky started \n")
    print("Leaky started ")
    
    global deposit_edge
    global backup_edge
    global cX, cY
    
    # Initialise cameras - hold auto white balance constant to improve filtering
    #stdscr.addstr("Setting camera gains and white balance \n")
    print("Setting camera gains and white balance ")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    os.system('v4l2-ctl --set-ctrl=white_balance_temperature_auto=0')
    os.system('v4l2-ctl --set-ctrl=white_balance_temperature=2800')
    os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
    os.system('v4l2-ctl --set-ctrl=exposure_absolute=150')
    os.system('v4l2-ctl --set-ctrl=brightness=0')    
    
    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:5:i')
    
    #stdscr.addstr("Establishing humidity sensing GPIO settings \n")
    print("Establishing humidity sensing GPIO settings ")
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
    
    hum_threshold = 70
    
    # Set up motors
    #stdscr.addstr("Setting motor parameters \n")
    print("Setting motor parameters")
    leaky1 = LeakyBot(myMotor1, myMotor2)    
    leaky1.speed = 140
    leaky1.direction = 'fwd'
    
    #stdscr.addstr("Setting navigation parameters \n")
    print("Setting navigation parameters")

    leaky1.threshold_state = 'leq'
    
    start_turning_frame = 0
    
    running = True
    
    fcount = 1
    ccount = 1

    winset = 0
    
    #stdscr.addstr("Waiting for block ... \n")
    print("Waiting for block ...")

    sense_delay = 1.5
    
    while running:    
        # STATE CHECK
        #print("State check", leaky1.direction, leaky1.cam_flag, leaky1.threshold_state, leaky1.state)
        
        # if we have windows open, use waitkey
        if winset:
            key = cv2.waitKey(1) & 0xFF
        #else:
            #key = stdscr.getch()
        
        # check for button pushes
        if leaky1.is_waiting() or leaky1.is_deposit():
            try:                
                block_trigger = 1.0 - block_pin.read()
                time.sleep(0.3)
                if block_trigger > 0.02:
                    #stdscr.addstr(str(block_trigger))
                    #stdscr.addstr(" \n")
                    print("Triggered, " , block_trigger)
                    leaky1.button_push()
                
            except Exception as e:
                print("Problem reading board, retry ...")
                #stdscr.addstr("problem reading board, retry ... \n")
                print(e)    
        
        # get camera frames
        right_frame = webcam.read()
        left_frame = picam.read()

        if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()) or (leaky1.is_go_home()):
            #stdscr.addstr("moving, no sensors \n")
            print("moving, no sensors")
            # set motor values
            if leaky1.is_go_home():
                if np.abs(home_match) > 3:
                    leaky1.set_motor_values(int(left_speed), int(right_speed))
                else:
                    leaky1.set_motor_values(leaky1.speed, leaky1.speed)

            elif leaky1.is_turning():
                leaky1.on_enter_turning()
                
            elif leaky1.is_deposit():
                leaky1.set_motor_values(leaky1.speed, leaky1.speed)
                                
            elif leaky1.is_backup():
                leaky1.on_enter_backup()
                            
            time.sleep(0.12)

            # Start sensing subloop
            leaky1.set_motor_values(0,0)
            #stdscr.addstr("Entering sensor loop \n")
            print("Entering sensor loop")
            start_time = time.time()

            while (time.time() - start_time < sense_delay):
                
                blue_walls_match, blue_left_wall, blue_right_wall = balance_detection(left_frame, right_frame, l_green_left, l_green_right, u_green_left, u_green_right, 550000.0, 580000.0, 0.0, 0.0)
                home_match, left_home, right_home  = balance_detection(left_frame, right_frame, l_red_left, l_red_right, u_red_left, u_red_right, 500000.0, 600000.0, 0.0, 0.0)

                if (leaky1.is_turning()):
                    # debug
                    if leaky1.high_humidity:
                        print("balancing blue walls: ", blue_walls_match, blue_left_wall, blue_right_wall)
                    # check blue wall balancing
                
                    if (blue_left_wall > 0.1) and (blue_right_wall > 0.1) and ccount > 1:
                        if (np.abs(blue_walls_match) < 4):
                            #stdscr.addstr("Walls balanced \n")
                            # debug
                            if leaky1.high_humidity: print("Walls balanced")
                            leaky1.walls_balanced()

                            # ensures each cycle starts with no edge found flag
                            deposit_edge = False
                            backup_edge = False

                elif (leaky1.is_backup() or leaky1.is_go_home()):
                    # check red wall balancing                    
                    print("Balancing red walls: ", home_match, left_home, right_home)

                    if (np.abs(home_match) < 1) and leaky1.is_backup() and (left_home > 0.5):
                        leaky1.home_spotted()
                    
                    elif leaky1.is_go_home():
                        print("checking homing size: ", left_home, right_home)
                        if (left_home > 12) or (right_home > 12):
                            #stdscr.addstr("I'm home! Waiting ... \n")
                            print("I'm home! Waiting ...")
                            leaky1.close_to_home()
                        
                        else: 
                            left_speed = leaky1.speed - 2*home_match
                            right_speed = leaky1.speed + 2*home_match


                if (leaky1.is_turning() and not leaky1.high_humidity) or leaky1.is_deposit():
                    #debug
                    if leaky1.cam_flag:
                        deposit_edge = find_centroid(left_frame, l_green_left, u_green_left, leaky1.threshold_state, leaky1.cam_flag, found_edge_left)
                        backup_edge = find_centroid(left_frame, l_green_left, u_green_left, leaky1.threshold_state, leaky1.cam_flag, back_up_left)
                        #deposit_edge = find_edge(left_frame, l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.cam_flag, 640-found_edge_threshold)
                        #backup_edge = find_edge(left_frame, l_blue_left, u_blue_left, leaky1.threshold_state, leaky1.cam_flag, 640-back_up_threshold)
                        cv2.circle(left_frame, (cX,cY), 7, (255,0,0), -1)

                    else:
                        deposit_edge = find_centroid(right_frame, l_green_right, u_green_right, leaky1.threshold_state, leaky1.cam_flag, found_edge_right)
                        backup_edge = find_centroid(right_frame, l_green_right, u_green_right, leaky1.threshold_state, leaky1.cam_flag, back_up_right)

                        #deposit_edge = find_edge(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.cam_flag, found_edge_threshold)
                        #backup_edge = find_edge(right_frame, l_blue_right, u_blue_right, leaky1.threshold_state, leaky1.cam_flag, back_up_threshold)
                        cv2.circle(right_frame, (cX,cY), 7, (255,0,0), -1)
            
                    #print("Looking for edge: ", cX)
                    # to-do (maybe): implement low-pass filter


                # handle whatever has been triggered, depending on state
                if (leaky1.is_turning()) and deposit_edge and not leaky1.high_humidity:
                    leaky1.wall_found()
                    backup_edge = False
                    
                elif leaky1.is_deposit() and backup_edge:
                    leaky1.reached_wall()
                    deposit_edge = False
                    backup_edge = False
                                                                    
                # loop a few times in case the vision is lagging
                # may need to hack the backup edge situation
                # now fall through

            blue_wm_prev = blue_walls_match
            ccount += 1


        if winset:
            cv2.imshow("Left camera", left_frame)
            cv2.imshow("Right camera", right_frame)
                                                
                        
        # CHECK HUMIDITY DATA
        # nB: reading from the humidity sensors is SLOW
        
        if leaky1.is_sensing():
            if (time.time() - leaky1.sensing_clock < 15):
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
                #stdscr.addstr("Entering final sensor read ...\n")
                print("Entering final sensor read ...")
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
                        #stdscr.addstr("continue ... \n")
                        print("continuing")

                    
                if hum_count > 0:
                    hum_av = hum_sum/hum_count
                    print("Average humidity: ", int(hum_av))
                    #stdscr.addstr(" \n")
                    
                    if hum_av < hum_threshold:
                        leaky1.high_humidity = False
                        leaky1.low_humidity()
                                
                    else:
                        # transitions to driving
                        leaky1.humidity_maintained()
                
                else:
                    #stdscr.addstr("no sensors available, starting again \n")
                    print("No sensors available, starting again")
                    leaky1.sensing_clock = time.time()

                

        elif leaky1.is_driving():
            if (time.time() - leaky1.driving_clock < 0.5):
                time.sleep(0.1)
            
            else:
                leaky1.stop_driving()
        
               
        fcount += 1
        
        if winset:
            if key == ord("q"):
                running=False
                break

        
    board.exit()
    webcam.stop()
    picam.stop()
    shutdownLeaky()
    

if __name__ == '__main__':
    main()
    #curses.wrapper(main)
    
