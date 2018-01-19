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

from LeakyBot import LeakyBot
from leaky_nav_functions import *

# Motor initialisation:
print("Initialising motor hat ...")
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)
print("...done")


print("Initialising cameras ...")
# Camera initialization
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
# ----------------------------------------------------------------------
cp = [ 300, 300]
r_out = 290;
r_inner = 145;
r_norim = 260;

poly_front = np.array([cp, [20, 1], [600,1]])
poly_back = np.array([cp, [130, 600],[1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])
# ----------------------------------------------------------------------
sides_mask, front_mask, wide_mask = define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

# ----------------------------------------------------------------------
lg_bound = 60
ug_bound = 100

# Initial guesses for filters
l_green = np.array([40, 60, 0])
u_green = np.array([90, 255, 255])
l_red = np.array([150, 80, 50])
u_red = np.array([180, 255, 255])
omni_frame = np.zeros((600,800,3))
# ----------------------------------------------------------------------


# shuts down motors and cameras on program exit, cleans up terminal
def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)

    picam.stop()
    cv2.destroyAllWindows()

atexit.register(shutdownLeaky)


def main():

    print("Leaky started ")
    
    global deposit_edge
    global backup_edge
    global cX, cY
    
    # Initialise cameras - hold auto white balance constant to improve filtering
    print("Setting camera gains and white balance ")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:5:i')
    
    print("Establishing humidity sensing GPIO settings ")
    # Set up humidity sensing
    gp.setwarnings(False)
    gp.setmode(gp.BCM)

    gp.setup(11, gp.OUT)
    gp.setup(17, gp.IN)
    gp.setup(27, gp.IN)
    gp.setup(22, gp.IN)
    
    # usage: ST(clock pin, data pin) (default voltage)
    sens1 = ST(11, 17)
    sens2 = ST(11, 27)
    sens3 = ST(11, 22)

    sens_array = [sens1, sens2, sens3]
    depvec = np.array([1, 1, 1, 1, 0, 0])
    homevec = np.array([0,0,0,0,1,1])
    
    hum_threshold = 65
    
    # Set up motors
    print("Setting motor parameters")
    leaky1 = LeakyBot(myMotor1, myMotor2)    
    leaky1.speed = 200
    leaky1.direction = 'fwd'
    
    print("Setting navigation parameters")

    # take a snapshot, make an adaptive estimate of colour filter
    init_frame = picam.read()
    init_crop = init_frame[367:937, 536:1136,:]
    # update boundary functions (assumes we have some blocks visible)
    l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound)
    
    running = True    
    fcount = 1
    kill_count = 1
    prepare_wait_flag = 0

    winset = 0
    
    print("Waiting for block ...")
    while running:    
        
        # check for button pushes
        if leaky1.is_waiting() or leaky1.is_deposit():
            try:                
                block_trigger = 1.0 -  block_pin.read()
                time.sleep(0.3)
                if block_trigger > 0.02:
                    print("Triggered, " , block_trigger)
                    leaky1.button_push()
                
            except Exception as e:
                print("Problem reading board, retry ...")
                print(e)    
        
        # get camera frames
        full_frame = picam.read()
        omni_frame = full_frame[367:967, 536:1136,:]
        save_frame = omni_frame.copy()
        
        try: prev_frame                
        except NameError: print("...") #no frame yet
        else: 
			if leaky1.direction == 'turn': # alternates
                diff_val = sim_mse(save_frame, prev_frame)
                

        if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()) or (leaky1.is_go_home()):
            print("Deposition probability: ", leaky1.probability)
	        # -----------------------------------
	        # Each loop, move in pre-set direction, then stop and sense
            print("moving, no sensors")
            leaky1.set_motor_values(leaky1.speed, leaky1.speed)
            time.sleep(0.14)

            # Start sensing subloop
            leaky1.set_motor_values(0,0)
            #print("Entering sensor loop")
            start_time = time.time()
            # -----------------------------------
            
            if (leaky1.is_turning()):
                # set direction
                leaky1.set_turn_direction()

                # we are either a) looking for wall balancing or b)looking to deposit
                if leaky1.high_humidity:
                    print("Balancing walls: ")
                    heading_angle, show_frame = omni_balance(cp, omni_frame, sides_mask, l_green, u_green)
                    rednum, red_head, _ , _ = omni_home(cp, omni_frame, sides_mask, l_red, u_red)
                    print(heading_angle)

                    # ---------------------------------
                    # make sure we cannot see home base, try to balance walls
                    if rednum > 0:
                        if red_head < 0: leaky1.cam_flag = 1
                        else: leaky1.cam_flag = 0

                    elif (heading_angle > 2.9) or (heading_angle < -2.9) :
                        print('walls balanced!')
                        leaky1.walls_balanced()

                    elif heading_angle > 0: leaky1.cam_flag = 1        
                    elif heading_angle < 0: leaky1.cam_flag = 0
                    else: print('no walls in view')
                    
                else: # we must be looking for a deposition spot
                    blob_num, heading_angle, box_ratio, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green)
                    
                    rednum, _, _ , _ = omni_home(cp, omni_frame, wide_mask, l_red, u_red)
                    #print('Looking for deposition, angle: ', heading_angle)
                    	
                    if ((heading_angle > 2.3) or (heading_angle < -2.3)) and rednum > 0:
                        print("moving to deposition")
                        leaky1.wall_found()
                    
                    # SET PROBABILITY
                    if blobnum > 0:
                        probvec = probability_calculator(blob_num, box_ratio)
                        leaky1.set_probability([probvec]) # should only set the relevant diagonals
                    
                    # CHECK SIM, UPDATE PROBABILITY:
                    # should also decrease probability if too many large differences build up
                    if diff_val < 80:
						simvec = np.array([0 0 0.1 0 0 0])
					elif diff_val > 800:
						simvec = np.array([0 0 -0.1 0 0 0 ])
        			else:
						simvec = np.zeros(6)
						
					leaky1.update_probability(simvec)
					scalar_prob = localisation_calculator(depvec, leaky1.probability)
					
                    # DO PROBABILITY CONDITIONALS

                    if scalar_prob > 0.9:
                        leaky1.static_visuals()
                        leaky1.set_probability([0,0,0,0]) # only sets deposition values

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/TurnForDepOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)

            elif leaky1.is_deposit():
                
                blob_num, heading_angle, box_ratio, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green)

                # SET PROBABILITY
                if blobnum > 0:
                    probvec = probability_calculator(blob_num, box_ratio)
                    leaky1.set_probability([probvec]) # should only set the relevant diagonals
                        
						
				leaky1.update_probability(simvec)

                if (leaky1.probability > 0.9) or (kill_count > 30): 
                    kill_count = 0
                    leaky1.set_probability([0, 0, 0, 0])
                    leaky1.reached_wall()

                if (heading_angle > 2.6) or (heading_angle < -2.6):
                    leaky1.direction = 'fwd'
                    #print("Heading straight there: ",  heading_angle*180/np.pi)

                else: # alternate directions when turning during deposition
                    if leaky1.direction == 'revturn':
                        leaky1.direction = 'turn'                        

                    else: 
                        # Only update similarity vector on forward turns
                        if diff_val < 80:
        				    simvec = np.array([0 0 0.1 0 0 0])
                	    else:
				            simvec = np.zeros(6)
				            
                        leaky1.direction = 'revturn'

                    if heading_angle < 0: leaky1.cam_flag = 0
                    elif heading_angle > 0: leaky1.cam_flag = 1
                
                kill_count += 1

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/DepOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)

            elif (leaky1.is_backup() or leaky1.is_go_home()):
                
                # look for red walls. Can we dynamically update the red filter in any useful way?
                red_locs, heading_angle, red_sizes, show_frame = omni_home(cp, omni_frame, wide_mask, l_red, u_red)

                print("Balancing red markers", red_sizes)
                if leaky1.is_backup():
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
                            #if leaky1.cam_flag: leaky1.generic_right_turn()
                            #else: leaky1.generic_left_turn()
                            
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
                        leaky1.set_turn_direction()
                        if heading_angle > 0: leaky1.cam_flag = 1
                        else: leaky1.cam_flag = 0

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
        if leaky1.is_sensing():
            if (time.time() - leaky1.sensing_clock < 30):
                time.sleep(0.1)

                hum_count = 0                
                for sens_i in sens_array:
                    try:
                        temp_new = sens_i.read_t()
                        hum_new = sens_i.read_rh()
                        hum_count +=1
                        
                    except Exception as e:
                        print("Sensor problem: ", (hum_count+1))

                #img = Image.fromarray(save_frame)
                #imname = './TestIm/SenseOutput_'
                #imname += str(fcount)
                #imname += '.png'
                #img.save(imname)
                
            else: # last sensor reading
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
                        print("Sensor problem: ", (hum_count+1), "continuing")

                    
                if hum_count > 0:
                    hum_av = hum_sum/hum_count
                    print("Average humidity: ", int(hum_av))
                    
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
    
