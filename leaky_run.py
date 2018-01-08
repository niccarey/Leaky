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
from PIL import Image
import threading 
import curses

from LeakyBot import LeakyBot
from leaky_nav_functions import *

# initialisation: any global variables, etc 

# Motor initialisation:
print("Initialising motor hat ...")
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)
print("...done")

print("Initialising cameras ...")
# Camera initialization
#webcam = VideoStream(src=0).start()
picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()

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

# Define omnicam masks:
cp = [ 389, 297]
r_out = 290;
r_inner = 145;
r_norim = 260;

poly_front = np.array([cp, [20, 1], [620,1]])
poly_back = np.array([cp, [1, 600], [800,600], [800,420]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [620, 1], [800 ,1], [800, 420]])

sides_mask, front_mask, wide_mask = define_masks(cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

lg_bound = 50
ug_bound = 110

l_green = np.array([40, 60, 0])
u_green = np.array([90, 255, 255])
l_red = np.array([150, 100, 30])
u_red = np.array([180, 255, 255])
omni_frame = np.zeros((600,800,3))

# shuts down motors and cameras on program exit, cleans up terminal
def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)

    picam.stop()
    cv2.destroyAllWindows()
    curses.endwin()


atexit.register(shutdownLeaky)

def update_similarity(robot, diff):
    if diff < 80:
        robot.similarity += 1
    else:
        robot.similarity = 0

    return robot

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
    
    hum_threshold = 78
    
    # Set up motors
    #stdscr.addstr("Setting motor parameters \n")
    print("Setting motor parameters")
    leaky1 = LeakyBot(myMotor1, myMotor2)    
    leaky1.speed = 200
    leaky1.direction = 'fwd'
    
    #stdscr.addstr("Setting navigation parameters \n")
    print("Setting navigation parameters")
    # take a snapshot, make an adaptive estimate of colour filter
    init_frame = picam.read()
    init_crop = init_frame[196:796, 432:1232,:]
    # update boundary functions (assumes we have some blocks visible)
    l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound)
    
    running = True
    
    fcount = 1
    kill_count = 1
    prepare_wait_flag = 0

    winset = 0
    
    #stdscr.addstr("Waiting for block ... \n")
    print("Waiting for block ...")

    while running:    
        # STATE CHECK
        #print("State check", leaky1.direction, leaky1.cam_flag, leaky1.threshold_state, leaky1.state)
        
        # if we have windows open, use waitkey
        
        # check for button pushes
        if leaky1.is_waiting() or leaky1.is_deposit():
            try:                
                block_trigger = 1.0 -  block_pin.read()
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
        full_frame = picam.read()
        omni_frame = full_frame[196:796, 432:1232,:]
        save_frame = omni_frame.copy()
        
        #else:
            #key = stdscr.getch()

        try: prev_frame
        
        except NameError: print("...") #no frame yet
        else: diff_val = sim_mse(save_frame, prev_frame)

        if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()) or (leaky1.is_go_home()):
            #stdscr.addstr("moving, no sensors \n")
            print("moving, no sensors")
            leaky1.set_motor_values(leaky1.speed, leaky1.speed)
            time.sleep(0.14)

            # Start sensing subloop
            leaky1.set_motor_values(0,0)
            #stdscr.addstr("Entering sensor loop \n")
            print("Entering sensor loop")
            start_time = time.time()

            if (leaky1.is_turning()):

                # alternates directions to a) avoid dripper jam and b) hopefully keep block in place
                if leaky1.direction == 'turn':
                    leaky1.direction = 'revturn'
                elif leaky1.direction == 'revturn':
                    leaky1.direction = 'turn'

                # we are either a) looking for wall balancing or b)looking to deposit
                if leaky1.high_humidity:
                    print("Balancing walls: ")
                    heading_angle, show_frame = omni_balance(cp, omni_frame, sides_mask, l_green, u_green)
                    rednum, red_head, _ , _ = omni_home(cp, omni_frame, sides_mask, l_red, u_red)
                    print(heading_angle)

                    if rednum > 0:
                        if red_head < 0:
                            leaky1.cam_flag = 1

                        else:
                            leaky1.cam_flag = 0

                    elif (heading_angle > 2.9) or (heading_angle < -2.9) :
                        print('walls balanced!')
                        leaky1.walls_balanced()

                    elif heading_angle > 0:
                        leaky1.cam_flag = 1
                        print("I think I'm turning left")
        
                    elif heading_angle < 0:
                        leaky1.cam_flag = 0
                        print("I think I'm turning right")

                    else: print('no walls in view')
                    
                    #img = Image.fromarray(show_frame)
                    #imname = './TestIm/BalanceOutput_'
                    #imname += str(fcount)
                    #imname += '.png'
                    #img.save(imname)
                
                else: # we must be looking for a deposition spot
                    heading_angle, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green)
                    rednum, _, _ , _ = omni_home(cp, omni_frame, wide_mask, l_red, u_red)
                    print('Looking for deposition, angle: ', heading_angle)
                    if ((heading_angle > 2.3) or (heading_angle < -2.3)) and rednum > 0:
                        leaky1.similarity = 0
                        leaky1.wall_found()

                    leaky1 = update_similarity(leaky1, diff_val)

                    if leaky1.similarity > 8:
                        leaky1.static_visuals()
                        leaky1.similarity = 0

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/TurnForDepOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)

            elif leaky1.is_deposit():
                leaky1 = update_similarity(leaky1, diff_val)
                #print kill_count

                if (leaky1.similarity > 3) or ((kill_count > 30) and (diff_val<150)): 
                    kill_count = 0
                    leaky1.reached_wall()
                    leaky1.similarity = 0

                heading_angle, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green)

                if (heading_angle > 2.6) or (heading_angle < -2.6):
                    leaky1.direction = 'fwd'
                    print("Heading straight there: ",  heading_angle*180/np.pi)

                else: # alternate directions when turning during deposition
                    if leaky1.direction == 'revturn':
                        leaky1.direction = 'turn'
                    else: 
                        leaky1.direction = 'revturn'

                    if heading_angle < 0:
                        leaky1.cam_flag = 0
                    elif heading_angle > 0:
                        leaky1.cam_flag = 1
                
    
                kill_count += 1

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/DepOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)

            elif (leaky1.is_backup() or leaky1.is_go_home()):
                
                # look for red walls
                red_locs, heading_angle, red_sizes, show_frame = omni_home(cp, omni_frame, wide_mask, l_red, u_red)

                print("Balancing red markers", red_sizes)
                if leaky1.is_backup():
					# is there a way to update the red filter? probably not
                    print("Markers seen: ", red_locs)
                    if (red_locs < 2):
                        print('home not found')
                        leaky1.direction='revturn'
                    
                    else:
                        leaky1.have_block = False
                        kill_count = 0 # failsafe                        
                        leaky1.home_spotted()

                elif leaky1.is_go_home():
                    if prepare_wait_flag:
                        if red_locs < 1:
                            prepare_wait_flag = 0
                            leaky1.close_to_home()
                        else:
                            if leaky1.cam_flag:
                                leaky1.generic_right_turn()
                            else:
                                leaky1.generic_left_turn()
                            
                            prepare_wait_flag = 0
                            leaky1.close_to_home()

                    elif red_locs < 1:
                        leaky1.direction='turn'
                        leaky1.cam_flag = 1

                    elif max(red_sizes) > 3000:
                        print("Preparing for new block")
                        prepare_wait_flag = 1
                        leaky1.cam_flag = bool(random.getrandbits(1))

                    elif (heading_angle > 2.3) or (heading_angle < -2.3):
                        print("Going fwd ...")
                        leaky1.direction='fwd'

                    else: 
                        if leaky1.direction == 'turn':
                            leaky1.direction = 'revturn'
                        else:
                            leaky1.direction = 'turn'

                        if heading_angle > 0:
                            leaky1.cam_flag = 1

                        else:
                            leaky1.cam_flag = 0

                    print("Heading angle: ", heading_angle*180/np.pi)
    

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/HomeOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)

        if winset:
            cv2.imshow("Camera view", show_frame)
            key = cv2.waitKey(1) & 0xFF
                

                        
        # CHECK HUMIDITY DATA
        # nB: reading from the humidity sensors is SLOW
        
        if leaky1.is_sensing():
            if (time.time() - leaky1.sensing_clock < 30):
                
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

                #img = Image.fromarray(save_frame)
                #imname = './TestIm/SenseOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)
                
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
            leaky1 = update_similarity(leaky1, diff_val)
            if leaky1.similarity > 5:
                leaky1.static_visuals()
                leaky1.similarity = 0

            elif (time.time() - leaky1.driving_clock < 0.5):
                time.sleep(0.1)
            
            else:
                leaky1.stop_driving()
        
        prev_frame = save_frame
        fcount += 1
        
        if winset:
            if key == ord("q"):
                running=False
                break

        
    board.exit()
    picam.stop()
    shutdownLeaky()
    

if __name__ == '__main__':
    main()
    
