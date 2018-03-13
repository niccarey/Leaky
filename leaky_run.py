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
#import threading 

from LeakyBot import LeakyBot
import leaky_nav_functions as lns
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
# Filtering constants (do I use these outside navigation??)
erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)


# Define omnicam masks:
# ----------------------------------------------------------------------
cp = [ 300, 300 ]
r_out = 298;
r_inner = 150;
r_norim = 295;

y_crop_min = 320
y_crop_max = 920

x_crop_min = 530
x_crop_max = 1130

poly_front = np.array([cp, [1, 20], [1,1], [600,1], [600,20]])
poly_back = np.array([cp, [1, 430],[1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])

# ----------------------------------------------------------------------
sides_mask, front_mask, wide_mask = lns.define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

# ----------------------------------------------------------------------
lg_bound = 42
ug_bound = 70

lr_bound = 0
ur_bound = 13

# Initial guesses for filters (overwritten later, can probably delete)
l_green = np.array([40, 60, 0])
u_green = np.array([90, 255, 255])
l_red = np.array([0, 80, 50])
u_red = np.array([20, 255, 255])
omni_frame = np.zeros((600,600,3))
# ----------------------------------------------------------------------

# shuts down motors and cameras on program exit, cleans up terminal
def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)
    picam.stop()
    cv2.destroyAllWindows()

atexit.register(shutdownLeaky)

def imgsave(image_array, imname, fcount):
    storeIm = Image.fromarray(image_array)
    imname += str(fcount)
    imname += '.jpg'
    storeIm.save(imname)

def flex_sensor_calib(flex1, flex2, vcc):
    flexcount = 0
    f1s = 0
    f2s = 0

    while (flexcount < 10):
        time.sleep(0.3)
        f1v = flex1.read()*vcc
        f2v = flex2.read()*vcc
        f1s += f1v
        f2s += f2v
        flexcount += 1

    return f1s/10, f2s/10


def main():
    print("Leaky started ")

    # Initialise cameras - hold auto white balance constant to improve filtering
    print("Setting camera gains and white balance ")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain

    # Set Arduino pressure sensor pin
    print("Setting up flex sensors")
    block_pin = board.get_pin('a:1:i')
    flex1 = board.get_pin('a:0:i')
    flex2 = board.get_pin('a:2:i')
    vcc = 3.3
    f1_av, f2_av = flex_sensor_calib(flex1, flex2, vcc)
    print("Whiskers: ", f1_av, f2_av)

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

    hum_threshold = 75
    
    print("Setting up electromagnet block holder")
    gp.setup(18, gp.OUT) 
    gp.output(18, 1) # high is off (thanks to inverting transistor)

    # Set up motors
    print("Setting motor parameters")
    leaky1 = LeakyBot(myMotor1, myMotor2)
    leaky1.speed = 170
    leaky1.direction = 'fwd'

    print("Setting up homing system")
    raw_input("Press Enter when ready for homing snapshot")
    
    # take a snapshot, make an adaptive estimate of colour filter and set homing snapshot
    init_frame = picam.read()
    init_crop = init_frame[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]
    # update boundary functions (assumes we have some blocks visible)
    print("Identify green peak:")
    l_green, u_green = lns.boundary_estimate(init_crop.copy(), lg_bound, ug_bound, 20, 255, 0, 255, 15)
    print("Identify red peak:")
    l_red, u_red = lns.boundary_estimate(init_crop.copy(), lr_bound, ur_bound, 90, 255, 180, 240, 10)
    print(" ... setting up unwarp map ...")
    xmap, ymap = lns.buildMap(600,600, 720, 360, 300, cp[0], cp[1])
    print("...done")
    
    print(" ... initialise tracking mask: ")
    o_width, tracking_mask, unwrap_gray = lns.init_tracking_mask(xmap, ymap, l_red, u_red, init_crop.copy(), wide_mask.copy())

    print(" ... establishing SIFT features")
    surf = cv2.xfeatures2d.SURF_create()
    kp_surf, des_surf = surf.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))
    
    # Image storage
    imdisp = cv2.drawKeypoints(unwrap_gray, kp_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    template_im = Image.fromarray(imdisp)
    imname ='./TestIm/TemplateImage_date.jpg'
    template_im.save(imname)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    print(" ... estimating identified feature locations")
    InitialKeypoints = KeypointExpanded(kp_surf)
    a0 = -188.44
    a2 = 0.0072
    a3 = -0.0000374
    a4 = 0.0000000887
    xS = 125
    InitialKeypoints = lns.keypoint_height_calc(InitialKeypoints, a0, a2, a3, a4, xS)
    print("Finished setting up homing system")

    running = True
    fcount = 1
    kill_count = 1

    winset = 0
    start_lk_time = time.time()
    
    print("Waiting for block ...")
    with open("./TestIm/070318_trial2.txt", "a") as storefile:
        
        while running:
            looptime = time.time()
            # check for button pushes
            
            if leaky1.is_waiting():
                try:
                    block_trigger = block_pin.read()
                    time.sleep(0.2)
                    if block_trigger > 0.02:
                        print("Triggered, " , block_trigger)
                        storefile.write("Block trigger: " + str(time.time()-start_lk_time) + "\n")
                        
                        # Turn on electromagnet
                        # do an internal PWM cycle while we turn away from the entrance
                        gp.output(18, 0) # set low (on)
                        # pick a random direction to start
                        leaky1.cam_flag = bool(random.getrandbits(1))
                        
                        # REPLACE THIS with something less derpy if electromagnet works
                        get_away_count = 0
                        while get_away_count < 12:
                            leaky1.generic_turn(0.1)
                            gp.output(18,1)
                            time.sleep(0.001)
                            gp.output(18, 0) # set low (on)
                            get_away_count += 1

                        leaky1.button_push()
    
                except Exception as e:
                    print("Problem reading board, retry ...")
                    print(e)
            
            # Get camera frames
            full_frame = picam.read()
            omni_frame = full_frame[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]

            # At some point I removed difference imaging - reinstigate?
            #save_frame = omni_frame.copy()
        
            if (leaky1.is_turning()) or (leaky1.is_deposit()) or (leaky1.is_backup()):
                
                # IF SPEED UP WORKS: take out sleep and 0,0 motor settings, we just run while we process
                # -----------------------------------
                # Each loop, move in pre-set direction, then stop and sense
                leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
                #time.sleep(0.1)
    
                #leaky1.auto_set_motor_values(0,0)
                # -----------------------------------
    
                if (leaky1.is_turning()):
                    # set direction
                    # REMOVE IF ELECTROMAGNET WORKS
                    #leaky1.set_turn_direction()
    
                    # we are either a) looking for wall balancing or b)looking to deposit

                    # pulse electromagnet off then on (v quickly)
                    gp.output(18,1)
                    time.sleep(0.0005)
                    gp.output(18, 0) # set low (on)

                    if leaky1.high_humidity:                        
                        #print("Balancing walls: ") # this is a bit janky - we end up with a strong bias
                        rednum = lns.leaving_home(cp, omni_frame, wide_mask, l_red, u_red)
                        # ---------------------------------
                        # make sure we cannot see home base, try to balance walls
                        if rednum > 0:
                            print("Balancing - can see red bars: ", rednum)
                            #just keep turning
                            
                        else:
                            heading_angle = lns.omni_balance(cp, omni_frame, sides_mask, l_green, u_green)
                            if (heading_angle > 2.75) or (heading_angle < -2.75) :
                                print('walls balanced!')
                                leaky1.walls_balanced()
    
                            elif heading_angle > 0: leaky1.cam_flag = 1
                            elif heading_angle < 0: leaky1.cam_flag = 0
                            else: print('no walls in view')    
    
                    else: # we must be looking for a deposition spot
                        blob_num, heading_angle, maxminbox, box_ratio = lns.omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)
    
                        if blob_num > 0 and ((min(maxminbox) < 220) or max(maxminbox > 560)):
                            print("Visual overlap. Moving to deposition")
                            leaky1.wall_found()
    
                elif leaky1.is_deposit(): # This won't guarantee the block is deposited on a wall, but it will try damn hard
                    gp.output(18,1)
                    time.sleep(0.0005)
                    gp.output(18, 0) # set low (on)

                    try:
                        f1 = 150*(flex1.read()*vcc -f1_av)
                        f2 = 100*(flex2.read()*vcc -f2_av)
    
                        print("bend sensing: ", f1, f2)
    
                        if abs(f1) >10 or abs(f2) > 10:
                            kill_count = 0
                            leaky1.set_probability([0.0,0,0,0,0.0, 0.0])
                            leaky1.reached_wall()
                            gp.output(18, 1) # Turn off electromagnet
                            continue
    
                    except: print("No data from flex sensors")
    
                    blob_num, heading_angle, minmaxbox, box_ratio = lns.omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)
    
                    # SET PROBABILITY
                    if blob_num > 0:
                        blob_vec = lns.dep_prob_calculator(blob_num, box_ratio, minmaxbox)
                        probvec = np.diagonal([leaky1.probability]).copy()
                        np.put(probvec, [0,1], blob_vec.astype(float))
                        leaky1.set_probability([probvec])
    
                    deposit_prob = lns.localisation_calculator(depvec, leaky1.probability)
                    print("green blobs, location estimation, box_ratio: ", blob_num,  deposit_prob, box_ratio)
    
                    if (deposit_prob > 0.7) or (kill_count > 30): 
                        kill_count = 0
                        leaky1.set_probability([0, 0, 0, 0, 0.0, 0.0])
                        leaky1.reached_wall()
                        continue
    
                    #if leaky1.direction == 'revturn' or leaky1.direction == 'fwd':
                    #    if diff_val < 80: simvec = np.array([0,0,0.1,0,0,0])
                    #    else: simvec = np.zeros(6)
                    #    leaky1.update_probability(simvec)
    
                    if blob_num < 2 and (abs(heading_angle) > 2.6):
                        leaky1.direction = 'fwd'
    
                    elif (blob_num>1) or (abs(heading_angle) < 2.6):
                        leaky1.direction = 'turn'
    
                        if abs(heading_angle < 2.6):
                            if heading_angle > 0: leaky1.cam_flag = 0
                            elif heading_angle < 0: leaky1.cam_flag = 1
    
                    # REMOVE IF ELECTROMAGNET WORKS
                    #leaky1.set_turn_direction()

                    print('I think I am turning: ',leaky1.cam_flag, leaky1.direction, heading_angle)
                    kill_count += 1
    
    
                elif (leaky1.is_backup()):
                    # we are just looking for rear wall, then we transition to homing algorithm
                    red_num = lns.omni_home(cp, omni_frame, front_mask, l_red, u_red)
    
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
    
                    c_width, delta, h_store, tracking_comp, unwrap_gray_comp, home_check, red_cent = lns.run_tracking_mask(xmap, ymap, l_red, u_red, compare_im.copy(), wide_mask.copy(), o_width)
                    #imgsave(tracking_comp, './TestIm/homing_check_', hcount)
    
                    if h_store > 40: height_weight = 0.3
                    elif h_store > 30: height_weight = 0.1    
                    else: height_weight = 0
    
                    try:
                        f1 = 150*(flex1.read()*vcc -f1_av)
                        f2 = 100*(flex2.read()*vcc -f2_av)
                        print("sensor bend: ", f1, f2)
    
                        if height_weight> 0 and (abs(f1) > 10 or abs(f2) > 10):
                            #print("sensor bend trigger: ", f1, f2)
                            homing = False
                            leaky1.close_to_home()
                            continue

                        elif abs(f1)>20 or abs(f2)>20:
                            homing= False
                            leaky1.close_to_home()
                            continue

                    except: print("No info from bend sensors")
    
                    kp_comp_surf, des_comp_surf = surf.detectAndCompute(unwrap_gray_comp, tracking_comp.astype(np.uint8))
                    #imdisp = cv2.drawKeypoints(unwrap_gray_comp, kp_comp_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    #imgsave(imdisp, './TestIm/HomingView_', hcount)
                    
                    if (not (des_comp_surf is None)) :
                        surf_matches = bf.match(des_surf, des_comp_surf) 
                        surf_matches = sorted(surf_matches, key= lambda x:x.distance)
                        
                        rotation, x_est, y_est = lns.est_egomotion(surf_matches, InitialKeypoints, kp_comp_surf)
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
                    
                        elif x_est < -5 or (red_cent < 380 and red_cent > 340):
                            leaky1.direction = 'fwd'
                            leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
                            time.sleep(0.2)
                            leaky1.auto_set_motor_values(0,0)
    
                        if (abs(rotation*180/np.pi) < 10) and (height_weight>0): direction_weight += 0.1
                        else: direction_weight = 0
            
                        print("Weightings: ", delta, h_store, ratio_weight, height_weight, direction_weight)
                        print("Navigation info: ", rotation*180/np.pi, x_est, y_est)
                        weight_array = np.array([ratio_weight, height_weight, direction_weight])
    
                        if (np.sum(weight_array)> 0.4 and (height_weight>0)):
                            homing = False
                            leaky1.close_to_home()
    
                    else:
                        print("cannot find relevant features, backing up")
                        leaky1.direction = 'revturn'
                        leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
                        time.sleep(0.1)
                        leaky1.auto_set_motor_values(0,0)
            
                    hcount += 1
            
            if winset:
                cv2.imshow("Camera view", show_frame)
                key = cv2.waitKey(1) & 0xFF                
    
                            
            # CHECK HUMIDITY DATA        
            if leaky1.is_sensing():
                if (time.time() - leaky1.sensing_clock < 35):
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
                            storefile.write(str(time.time() - start_lk_time) + "Humidity: " + str(hum_new) + ", Temperature: "+ str(temp_new) + "\n")
                            hum_sum = hum_sum + hum_new
                            hum_count += 1
                            
                        except Exception as e:
                            print("Sensor problem: ", (hum_count+1), "continuing")
    
                        
                    if hum_count > 0:
                        hum_av = hum_sum/hum_count
                        print("Average humidity: ", int(hum_av))
                        
                        if hum_av < hum_threshold:
                            leaky1.high_humidity = False
                            storefile.write("New deposition: " + str(time.time() - start_lk_time) + "\n")
                            leaky1.low_humidity()
                                    
                        else:
                            # transitions to driving
                            storefile.write("Driving ... \n")
                            leaky1.humidity_maintained()
                    
                    else:
                        print("No sensors available, starting again")
                        leaky1.sensing_clock = time.time()
    
                #datfile.close()
    
            elif leaky1.is_driving():
                if (time.time() - leaky1.driving_clock < 0.2):
                    time.sleep(0.1)
                
                else: leaky1.stop_driving()
            
            #prev_frame = save_frame
            fcount += 1
            
            if winset:
                if key == ord("q"):
                    running=False
                    break
            print(time.time() - looptime)

        
    board.exit()
    picam.stop()
    shutdownLeaky()
    

if __name__ == '__main__':
    main()
    
