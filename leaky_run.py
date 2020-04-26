#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Structure and outline 

from picamera import PiCamera
from picamera.array import PiRGBArray
import imutils
from imutils.video import VideoStream
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from navImage import NavImage

import logging
import adafruit_tca9548a
import RPi.GPIO as gp
import Adafruit_GPIO.I2C as I2C
import Adafruit_ADS1x15

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
import kernel_def

logging.basicConfig(level=logging.DEBUG)
gp.setmode(gp.BCM)
gp.setwarnings(False)

# Initialise TCA (easier to do this before importing SHT C library)
error = 1
while error:
	try:
		I2CBusNum = 1
		tca_bus = I2C.Device(0x70, I2CBusNum)
		error = 0

	except:
		print("Trouble communicating with TCA multiplexer, trying again ...")

from sht_lib import SHT85_I2C
import SHTConstants as sensing_clock


# Initialise motors, including servo
logging.debug("Initialising motor hat ...")
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(2)
myMotor2 = mh.getMotor(4)
logging.debug("...done")

# Any interfacing using gp may need several attempts 
logging.debug("Setting up servo motor ...")
servoPin = 12
for attempt in range(0,2):
	try:
		gp.setup(servoPin, gp.OUT)
		servo_drive = gp.PWM(servoPin, 100)
		logging.debug("...done")
	except:
		print("GPIO communication problem, trying again ...")

# Initialise reflectance sensor, prox sensor, and flex sensor channels
refPin = 25
proxPin = 23
left_flex_chan = 0
right_flex_chan = 1
prox_chan = 3

# Initialise cameras
logging.debug("Setting up camera streaming ...")
picam = VideoStream(usePiCamera=True, resolution=(1640,1232)).start()
time.sleep(0.1)
logging.debug("...done")

# Initialise image processing tools, filters and masks
# Mask parameters are defined in kernel_def
sides_mask, front_mask, wide_mask = lns.define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)
    picam.stop()
    cv2.destroyAllWindows()
    for attempt in range(3):
    	try:
    		gp.cleanup()
    	except: 
    		print("Problems cleaning up, retry ...")


atexit.register(shutdownLeaky)

def imgsave(image_array, imname, fcount):
    storeIm = Image.fromarray(image_array)
    imname += str(fcount)
    imname += '.jpg'
    storeIm.save(imname)

def flex_sensor_calib(adc, flex1, flex2, gain):
    flexcount = 0
    f1s = 0
    f2s = 0

    while (flexcount < 10):
        time.sleep(0.05)
        f1v = lns.get_adc_reading(adc, flex1, gain)#flex1.read()*vcc
        f2v = lns.get_adc_reading(adc, flex2, gain)#flex2.read()*vcc
        f1s += f1v
        f2s += f2v
        flexcount += 1

    return f1s/10, f2s/10

def main():

    logging.debug("Begin deposition program")

	# Set camera gain and white balance
	logging.debug("Setting camera gains and white balance ...")
    camgain = (1.4,2.1)
    picam.camera.awb_mode = 'off'
    picam.camera.awb_gains = camgain
    logging.debug("... Done")

    # Initialise TCA channels
    adc_chan = 0B0000100

    # Initialise thresholds
    hum_threshold = 75 # humidity
    block_threshold = 50 # block placement threshold
    prox_threshold = 870 # homing proximity threshold
    PROXGAIN = 1

    hsens1 = 0B0000001
    hsens2 = 0B0000010
    hsens3 = 0B1000000

    humidity_channels = [hsens1, hsens2, hsens3]

    # Initialise humidity sensors, check if all are working
    for channel in humidity_channels:
    	for attempt in range(2):
			try: 
				tca_bus.writeRaw8(channel)
			except:
				print("Trouble moving to TCA device, retry ...")

		sht_instance = SHT85_I2C()
		tiime.sleep(0.02)
		serialNum, error = sht_instance.SHT_ReadSerial()
		if serialNum == 0:
			logging.debug("Trouble accessing sensor at ", channel)

		time.sleep(0.2)
		# clear flags
		error = sht_instance.SHT85_ClearFlags()
		time.sleep(0.2)


	# Initialise ADC, initialise and calibrate flex sensors
	logging.debug("Initialising flex sensors ...")

	# ADC goes through TCA, need to switch to correct channel
	for attempt in range(3):
		try: 
			tca_bus.writeRaw8(adc_chan)
		except:
			print("Trouble communicating with TCA multiplexer, trying again ...")


	adc = Adafruit_ADS1x15.ADS1115()
	FLEXGAIN = 1
	vcc = 5
	f1_av, f2_av = flex_sensor_calib(adc, left_flex_chan, right_flex_chan, FLEXGAIN, vcc)
    print("Whisker default positions: ", f1_av, f2_av)
    logging.debug("Done")


	# Initialise robot, incl. motors
	logging.debug("Setting up Leaky instance ...")
    leaky = LeakyBot(myMotor1, myMotor2, servo_drive, servoPin, 0.3)
    leaky.speed = 170
    leaky.direction = 'fwd'
    logging.debug("...Done")
	# Initialise servo motor position (make sure holder is UP)
    leaky.open_block_holder()


    # set up proximity sensor active line
    gp.setup(proxPin, gp.OUT)

	# Set up adaptive image thresholding, warp maps and any remaining masks
	logging.debug("Initialising HSV thresholding ... ")
	init_frame = picam.read()
    init_crop = init_frame[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]
    # update boundary functions (assumes we have some blocks visible)
    print("Identify green peak:")
    l_green, u_green = lns.boundary_estimate(init_crop.copy(), lg_bound, ug_bound, 20, 255, 0, 255, 15)
    print("Identify red peak:")
    l_red, u_red = lns.boundary_estimate(init_crop.copy(), lr_bound, ur_bound, 90, 255, 180, 240, 10)
    logging.debug(" ... setting up unwarp map ...")
    xmap, ymap = lns.buildMap(600,600, 720, 360, 300, cp[0], cp[1])

    print(" ... initialise tracking mask: ")
    o_width, tracking_mask, unwrap_gray = lns.init_tracking_mask(xmap, ymap, l_red, u_red, init_crop.copy(), wide_mask.copy())

    logging.debug("...Done")

	# Set RUNFlAG to true
	RUNFLAG = True
	# fcount = 1
    # kill_count = 1
	
	# set up file:
	date = str(221219)
	trial = "_trial1.txt"
	logname = "./TestIm/" + date + trial

	# Set experiment time to current time
    winset = 0
    start_lk_time = time.time()
    
    print("Waiting for block ...")

	# Open text file to write events and timing
    with open(logname, "a") as storefile:

    	while RUNFLAG:
    		looptime = time.time()
			# Wait for block: Try/Except loop for reflectance sensor

			if leaky1.is_waiting():
                try:
                	block_trigger = lns.get_block_reading(refPin)
                    if block_trigger < 50:
                        print("Triggered, " , block_trigger)
                        storefile.write("Block trigger: " + str(time.time()-start_lk_time) + "\n")
                        
                        # pick a random direction, close block holder, turn around
                        leaky.set_turn_status()
                        leaky.block_sensed()
                        time.sleep(0.1)
                        leaky.generic_turn(1.2) # this needs to be about 180


			# Get camera frame
            full_frame = picam.read()
            omni_frame = full_frame[y_crop_min:y_crop_max, x_crop_min:x_crop_max,:]


            if (leaky.is_turning()) or (leaky.is_deposit()) or (leaky.is_backup()):

                # Eac loop, move in pre-set direction, then stop and run appropriate sensing subfunction
                leaky.auto_set_motor_values(leaky.speed, leaky.speed)

                # ------ SENSOR SUBFUNCTIONS ------ #
                if (leaky.is_turning()):

                	if leaky.high_humidity:    

                        rednum = lns.leaving_home(cp, omni_frame, wide_mask, l_red, u_red)
                        # ---------------------------------
                        # make sure we cannot see home base, try to balance walls
                        if rednum > 0:
                            logging.debug("Balancing - can see red bars: ", rednum)
                            #just keep turning
                            
                        else:
                            heading_angle = lns.omni_balance(cp, omni_frame, sides_mask, l_green, u_green)
                            if (heading_angle > 2.75) or (heading_angle < -2.75) :
                                logging.debug('walls balanced!')
                                leaky.walls_balanced()
    
                            elif heading_angle > 0: leaky.cam_flag = 1
                            elif heading_angle < 0: leaky.cam_flag = 0
                            else: logging.debug('no walls in view') 
					
					else: # we must be looking for a deposition spot
                        blob_num, heading_angle, maxminbox, box_ratio = lns.omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)
    
                        if blob_num > 0 and ((min(maxminbox) < 220) or max(maxminbox > 560)):
                            logging.debug("Visual overlap. Moving to deposition")
                            leaky.wall_found()


                elif leaky.is_deposit(): # This won't guarantee the block is deposited on a wall, but it will try damn hard
                    blob_exists = 0
                    bend_exists = 0
                    ratio_exists = 0

                    # make sure appropriate channel is selected
                   	for attempt in range(3):
						try: 
							tca_bus.writeRaw8(adc_chan)
						except:
							print("Trouble communicating with TCA multiplexer, trying again ...")

					# Try flex sensor readings: F1 is obstructed by block holder, look at F2 only
					f2_read = 0
					bend_av_counter = 0
					for flexloop in range(4):
	                    try:
        	                f2 = lns.get_adc_reading(adc, right_flex_chan, FLEXGAIN)
        	                f2_read += f2
        	                bend_av_counter += 1
        	                time.sleep(0.005)

        	            except: 
        	            	print("problems logging flex sensor data ...")

        	        if bend_av_counter > 0:
                        f2_read /= bend_av_counter
                        f2_read = f2_read - f2_av
            	        logging.debug("bend sensing: ", f2_read)
    

                    blob_num, heading_angle, minmaxbox, box_ratio = lns.omni_deposit(cp, omni_frame, wide_mask, l_green, u_green, xmap, ymap)
    
    				# Set feature detector flags and call localisation calculator
    				if blob_num > 0: blob_exists = 1
    				if abs(f2_read) > 20: bend_exists = 1
    				if box_ratio > 1.5: ratio_exists = 1

    				new_prob = lns.localisation_prob(leaky.prev_prob, bend_exist, ratio_exist, walls_exist):
    				leaky.prev_prob = new_prob

    				if (new_prob > 0.75): 
                        leaky.reached_wall()
                        continue

                    if blob_num < 2 and (abs(heading_angle) > 2.6):
                        leaky.direction = 'fwd'
    
                    elif (blob_num>1) or (abs(heading_angle) < 2.6):
                        leaky.direction = 'turn'
    
                        if abs(heading_angle < 2.6):
                            if heading_angle > 0: leaky.cam_flag = 0
                            elif heading_angle < 0: leaky.cam_flag = 1
    
                    logging.debug('Direction update: ',leaky.cam_flag, leaky.direction, heading_angle)

                elif (leaky.is_backup()):
                    # we are just looking for rear wall, then we transition to homing algorithm
                    red_num = lns.omni_home(cp, omni_frame, front_mask, l_red, u_red)
    
                    logging.debug("Looking for entrance, Markers seen: ", red_num)
                    if (red_num < 2):
                        leaky.direction='revturn'
    
                    else:
                        leaky.have_block = False
                        kill_count = 0 # failsafe, shouldn't need this
                        leaky.home_spotted()


            elif leaky.is_go_home():
                homing = True
    
                while homing:
                	# Check proximity value:
                	gp.output(proxPin, True)
                	time.sleep(0.04)
                	gp.output(proxPin, False)

                	# make sure we are reading from adc channel (shouldn't have changed, but better safe)
                	for attempt in range(3):
						try: 
							tca_bus.writeRaw8(adc_chan)
						except:
							print("Trouble communicating with TCA multiplexer, trying again ...")

					for i in range(3):
						# crappy averaging filter to avoid jumps - can also use lowpass filter in lns
						close = 0
	                	distance = lns.get_adc_reading(adc, prox_chan, PROXGAIN)
	                	if distance > 900:
	                		close += 1                	

                    else: 
                    	red_num, red_centre, red_frame = lns.omni_home(cp, omni_frame, front_mask, l_red, u_red)
	                    # set direction:
    	                lns.homing_direction(red_cent)

    	                # turn or move forward (don't turn too much)
    	                if leaky.direction=='fwd':
    	                	if close > 2:
		                		logging.debug("Close to wall, stopping")
        		        		homing = False
                				leaky.close_to_home()
		                    
		                    else:
			                    leaky.auto_set_motor_values(leaky.speed, leaky.speed)
    			                time.sleep(0.2)
        			            leaky.auto_set_motor_values(0,0)
            
                  		else:
                  			leaky.auto_set_motor_values(leaky.speed, leaky.speed)
                  			time.sleep(0.08)
                  			leaky.auto_set_motor_values(0,0)

            if winset:
                cv2.imshow("Camera view", show_frame)
                key = cv2.waitKey(1) & 0xFF 

			if leaky.is_sensing():
				# Cycle through multiplexer channels:

				hum_av = 0
				sens_count = 0

				# Could put this in subfunction maybe
				for channel in humidity_channels:
					for attempt in range(2):
						try: 
							tca_bus.writeRaw8(channel)
						except:
							print("Trouble moving to TCA device, retry ...")

					sht_instance = SHT85_I2C()
					error = sht_instance.StartPeriodicMeasurement(sc.PERI_MEAS_HIGH_1_HZ)
					logging.debug(error)

					loop = 3
					hum = 0
					while loop > 0:
						# read a couple of times, keep last value
						temp, hum, error = sht_instance.SHT85_ReadBuffer()
						logging.debug("Humidity reading: ", hum)
						time.sleep(1)
						loop -= 1

					error = sht_instance.StopPeriodicMeasurement()

					if hum > 0:
					hum_av += hum
					sens_count += 1

				# after all sensors have been checked:
				if sens_count > 0:
					hum_av /= sens_count
					logging.debug("Average humidity reading: ", hum_av)
					logging.debug("Sensors active: ", sens_count)
                else:
                    print("No sensors available, starting again")
                    leaky.sensing_clock = time.time()

                if (time.time() - leaky1.sensing_clock > 10):
                	# check threshold, move or change state:
                    if hum_av < hum_threshold:
        	            leaky.high_humidity = False
        	            storefile.write("Humidity reading: " + str(hum_av))
            	        storefile.write("New deposition: " + str(time.time() - start_lk_time) + "\n")
 	                    leaky.low_humidity()
                	else:
                		storefile.write("Humidity reading: " + str(hum_av))
                        storefile.write("Driving ... \n")
                        leaky.humidity_maintained()
                    
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


	# Shutdown Leaky
	logging.debug("Finished deposition cycle")
    shutdownLeaky()
    

if __name__ == '__main__':
	main()
