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
from KeypointExpanded import KeypointExpanded

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
cp = [ 300, 300 ]
r_out = 298;
r_inner = 150;
r_norim = 295;
y_crop_min = 328
y_crop_max = 928
x_crop_min = 530
x_crop_max = 1130

poly_front = np.array([cp, [20, 1], [600,1]])
poly_back = np.array([cp, [130, 600],[1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])
# ----------------------------------------------------------------------
sides_mask, front_mask, wide_mask = define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

# ----------------------------------------------------------------------
lg_bound = 60
ug_bound = 100

lr_bound = 5
ur_bound = 25

# Initial guesses for filters (overwritten later, can probably delete)
l_green = np.array([40, 60, 0])
u_green = np.array([90, 255, 255])
l_red = np.array([0, 80, 50])
u_red = np.array([0, 255, 255])
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

    global cX, cY

    # Initialise cameras - hold auto white balance constant to improve filtering
    print("Setting camera gains and white balance ")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    # Set Arduino pressure sensor pin
    block_pin = board.get_pin('a:1:i')
    flex1 = board.get_pin('a:0:i')
    flex2 = board.get_pin('a:2:i')

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
    depvec = np.array([1.0, 1.0, 1.0, 1.0, 0, 0])
    homevec = np.array([0,0,0,0,1.0,1.0])

    hum_threshold = 65

    # Set up motors
    print("Setting motor parameters")
    leaky1 = LeakyBot(myMotor1, myMotor2)
    leaky1.speed = 160
    leaky1.direction = 'fwd'

    print("Setting up homing system")
    raw_input("Press Enter when ready for homing snapshot")
    
    # take a snapshot, make an adaptive estimate of colour filter and set homing snapshot
    init_frame = picam.read()
    init_crop = init_frame[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]
    # update boundary functions (assumes we have some blocks visible)
    print("Identify green peak:")
    l_green, u_green = boundary_estimate(init_crop.copy(), lg_bound, ug_bound, 50, 255, 0, 255, 20)
    print("Identify red peak:")
    l_red, u_red = boundary_estimate(init_crop.copy(), lr_bound, ur_bound, 100, 255, 100, 255, 15)
    # should I save??
    print(" ... setting up unwarp map ...")
    xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])
    print("...done")
    
    print(" ... initialise tracking mask: ")
    o_width, tracking_mask, unwrap_gray = init_tracking_mask(xmap, ymap, l_red, u_red, init_crop.copy(), wide_mask.copy())

    print(" ... establishing SIFT features")
    sift = cv2.xfeatures2d.SIFT_create()
    kp_sift, des_sift = sift.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

    imdisp = cv2.drawKeypoints(unwrap_gray, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    template_im = Image.fromarray(imdisp)
    imname ='./TestIm/TemplateImage_date.jpg'
    template_im.save(imname)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    print(" ... estimating identified feature locations")
    InitialKeypoints = KeypointExpanded(kp_sift)
    a0 = -188.44
    a2 = 0.0072
    a3 = -0.0000374
    a4 = 0.0000000887
    xS = 125
    InitialKeypoints = keypoint_height_calc(InitialKeypoints, a0, a2, a3, a4, xS)
    print("Finished setting up homing system")

    running = True
    fcount = 1
    kill_count = 1

    winset = 0

    print("Waiting for block ...")
    while running:

        # check for button pushes
        if leaky1.is_waiting() or leaky1.is_deposit() or leaky1.is_turning():
            try:
                block_trigger = block_pin.read()
                time.sleep(0.3)
                if block_trigger > 0.02:
                    print("Triggered, " , block_trigger)
                    leaky1.generic_rev_turn(0.12)
                    time.sleep(0.3)
                    leaky1.generic_turn(0.12)
                    time.sleep(0.3)
                    leaky1.generic_rev_turn(0.12)
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
            if (leaky1.direction == 'turn' or leaky1.direction=='fwd') and (not leaky1.is_sensing()):
                diff_val = sim_mse(save_frame, prev_frame)
                #print("vis sim: ", diff_val)


        if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()):
            # -----------------------------------
            # Each loop, move in pre-set direction, then stop and sense
            #print("moving, no sensors")
            leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
            time.sleep(0.1)

            # Start sensing subloop
            leaky1.auto_set_motor_values(0,0)
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
                    rednum, red_head, red_frame = leaving_home(cp, omni_frame, wide_mask, l_red, u_red)

                    # ---------------------------------
                    # make sure we cannot see home base, try to balance walls
                    print("Can see red bars: ", rednum)
                    if rednum > 0:
                        if red_head < 0: leaky1.cam_flag = 1
                        else: leaky1.cam_flag = 0

                    elif (heading_angle > 2.9) or (heading_angle < -2.9) :
                        print('walls balanced!')
                        leaky1.walls_balanced()

                    elif heading_angle > 0: leaky1.cam_flag = 1
                    elif heading_angle < 0: leaky1.cam_flag = 0
                    else: print('no walls in view')

                    #img = Image.fromarray(red_frame)
                    #imname = './TestIm/LeavingHome_'
                    #imname += str(fcount)
                    #imname += '.png'
                    #img.save(imname)

                else: # we must be looking for a deposition spot
                    blob_num, heading_angle, maxminbox, box_ratio, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)
                    #rednum, _, _   = leaving_home(cp, omni_frame, wide_mask, l_red, u_red)
                    print(maxminbox)
                    if blob_num > 0 and ((min(maxminbox) < 250) or max(maxminbox > 500)):
                        print("Visual overlap. Moving to deposition")
                        leaky1.wall_found()

                    #if (blob_num < 3) and ((heading_angle > 2.3) or (heading_angle < -2.3)) and rednum > 0:
                    #    print("moving to deposition")
                    #    leaky1.wall_found()

                    # SET PROBABILITY
                    #if blob_num > 0:
                    #    print('checking deposition targets:', blob_num, heading_angle, diff_val)
                    #    blob_vec = dep_prob_calculator(blob_num, box_ratio)
                    #    probvec = np.diagonal([leaky1.probability]).copy()
                    #    np.put(probvec,[0,1], blob_vec.astype(float))
                    #    leaky1.set_probability([probvec]) # should only set the relevant diagonals

                    #if diff_val < 80:
                    #    simvec = np.array([0.0, 0, 0.1, 0, 0, 0])
                    #else:
                    #    simvec = np.zeros(6).astype(float)

                    #try: leaky1.update_probability(simvec)
                    #except: print("Similarity estimate not found, continuing ...")
                    #deposit_prob = localisation_calculator(depvec, leaky1.probability)

                    #print("location estimation: ", deposit_prob)

                    #if deposit_prob > 0.8:
                    #    leaky1.static_visuals()
                    #    leaky1.set_probability([0.0,0,0,0,0,0.0]) # only sets deposition values

                #img = Image.fromarray(omni_frame)
                #imname = './TestIm/TurnForDepOutput_'
                #imname += str(fcount)
                #imname += '.jpg'
                #img.save(imname)

            elif leaky1.is_deposit():
                try:
                    f1 = 180 - 1000*flex1.read()
                    f2 = 180 - 1000*flex2.read()
                    print("bend sensing: ", f1, f2)

                    if abs(f1) >15 or abs(f2) > 15:
                        kill_count = 0
                        leaky1.set_probability([0.0,0,0,0,0.0, 0.0])
                        leaky1.reached_wall()
                        continue

                except: print("No data from flex sensors")
                blob_num, heading_angle, minmaxbox, box_ratio, show_frame = omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)

                # SET PROBABILITY
                if blob_num > 0:
                    blob_vec = dep_prob_calculator(blob_num, box_ratio, minmaxbox)
                    probvec = np.diagonal([leaky1.probability]).copy()
                    np.put(probvec, [0,1], blob_vec.astype(float))
                    leaky1.set_probability([probvec])

                deposit_prob = localisation_calculator(depvec, leaky1.probability)
                print("green blobs, location estimation: ", blob_num,  deposit_prob)

                if (deposit_prob > 0.8) or (kill_count > 30): 
                    kill_count = 0
                    leaky1.set_probability([0, 0, 0, 0, 0.0, 0.0])
                    leaky1.reached_wall()
                    continue

                if leaky1.direction == 'revturn' or leaky1.direction == 'fwd':
                    if diff_val < 80: simvec = np.array([0,0,0.1,0,0,0])
                    else: simvec = np.zeros(6)
                    leaky1.update_probability(simvec)

                if blob_num < 2 and (abs(heading_angle) > 2.6):
                    leaky1.direction = 'fwd'

                elif (blob_num>1) or (abs(heading_angle) < 2.6):
                    if leaky1.direction == 'fwd': leaky1.direction = 'turn'

                    if abs(heading_angle < 2.6):
                        if heading_angle > 0: leaky1.cam_flag = 0
                        elif heading_angle < 0: leaky1.cam_flag = 1

                leaky1.set_turn_direction()
                print('I think I am turning: ',leaky1.cam_flag, leaky1.direction, heading_angle, diff_val)
                kill_count += 1

                #img = Image.fromarray(show_frame)
                #imname = './TestIm/DepOutput_'
                #imname += str(fcount)
                #imname += '.jpg'
                #img.save(imname)

            elif (leaky1.is_backup()):
                # we are just looking for rear wall, then we transition to homing algorithm
                red_num, heading_angle, red_sizes, matched_filt_flag, show_frame = omni_home(cp, omni_frame, front_mask, l_red, u_red, l_green, u_green)

                print("Looking for entrance, Markers seen: ", red_num)
                if (red_num < 2):
                    leaky1.direction='revturn'

                else:
                    leaky1.have_block = False
                    kill_count = 0 # failsafe
                    leaky1.home_spotted()
            
        elif leaky1.is_go_home():
            homing = True
            direction_weight = 0
            ratio_weight = 0
            hcount = 0

            while homing:
                read_im = picam.read()
                compare_im = read_im[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]

                c_width, delta, h_store, tracking_comp, unwrap_gray_comp, home_check = run_tracking_mask(xmap, ymap, l_red, u_red, compare_im.copy(), wide_mask.copy(), o_width)
		
                if delta < 0.9: ratio_weight = 0
                elif delta > 0.9:
                    if ratio_weight < 0.2: ratio_weight = 0.2
                    else:  ratio_weight += 0.1

                if h_store > 60: 
                    height_weight = 0.3
                    try: 
                        f1 = 180-1000*flex1.read()
                        f2 = 180 - 1000*flex2.read()
                        if abs(f1) > 20 or abs(f2) > 20:
                            homing = False
                            leaky1.close_to_home()
                            continue
                    
                    except: print("No info from bend sensors")

                else: height_weight = 0

                kp_comp_sift, des_comp_sift = sift.detectAndCompute(unwrap_gray_comp, tracking_comp.astype(np.uint8))
                imdisp = cv2.drawKeypoints(unwrap_gray_comp, kp_comp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                #matchim = Image.fromarray(imdisp)
                #imname = './TestIm/template_matching_'
                #imname += str(hcount)
                #imname += '.jpg'
                #matchim.save(imname)
                #hcount += 1

                if (not (des_comp_sift is None)) :
                    sift_matches = bf.match(des_sift, des_comp_sift) 
                    sift_matches = sorted(sift_matches, key= lambda x:x.distance)
                    
                    rotation, x_est, y_est = est_egomotion(sift_matches, InitialKeypoints, kp_comp_sift)
                    if abs(rotation*180/np.pi) > 5:
                        direction_weight = 0
                        if rotation > 0:
                            leaky1.cam_flag = 1
                            leaky1.direction = 'left'
                            leaky1.set_motor_values(leaky1.speed, leaky1.speed, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.FORWARD)
                            time.sleep(0.08)
                            leaky1.auto_set_motor_values(0,0)
                
                        elif rotation < 0:
                            leaky1.cam_flag = 0
                            leaky1.direction = 'right'
                            leaky1.set_motor_values(leaky1.speed, leaky1.speed, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.BACKWARD)
                            time.sleep(0.08)
                            leaky1.auto_set_motor_values(0,0)
                
                    elif x_est < -5:
                        leaky1.direction = 'fwd'
                        leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
                        time.sleep(0.2)
                        leaky1.auto_set_motor_values(0,0)
              
                    if (abs(rotation*180/np.pi) < 10) and ((ratio_weight > 0) or (height_weight>0)): direction_weight += 0.1
                    else: direction_weight = 0
        
                    print("Weightings: ", delta, h_store, ratio_weight, height_weight, direction_weight)
                    print("Navigation info: ", rotation*180/np.pi, x_est, y_est)
                    weight_array = np.array([ratio_weight, height_weight, direction_weight])

                    if (np.sum(weight_array)> 0.5 and (ratio_weight >0 and height_weight>0)):
                        leaky1.cam_flag = bool(random.getrandbits(1))
                        homing = False
                        leaky1.close_to_home()
             		                
                else: 
                    print("cannot find relevant features, backing up")
                    #leaky1.cam_flag = 1
                    leaky1.direction = 'revturn'
                    leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
                    time.sleep(0.1)
                    leaky1.auto_set_motor_values(0,0)
        

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
            #leaky1 = update_similarity(leaky1, diff_val)
            #if leaky1.similarity > 5:
            #    leaky1.static_visuals()
            #    leaky1.similarity = 0

            if (time.time() - leaky1.driving_clock < 0.3):
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
    
