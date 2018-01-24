from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

from PIL import Image
import cv2
import imutils
from imutils.video import VideoStream
from navImage import NavImage
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import time
import atexit

from leaky_nav_functions import *


def imgsave(image_array, imname, fcount):
    storeIm = Image.fromarray(image_array)
    imname += str(fcount)
    imname += '.png'
    storeIm.save(imname)


# Define omnicam masks:
cp = [ 300, 300 ]
r_out = 298;
r_inner = 145;
r_norim = 260;

poly_front = np.array([cp, [20, 1], [600,1]])
poly_back = np.array([cp, [1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])

sides_mask, front_mask, wide_mask = define_masks([600,600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

lg_bound = 60
ug_bound = 110

lr_bound = 0
ur_bound = 10

picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()

time.sleep(0.5)

# To start:
# we want to tune the red filter like the green filter

init_frame = picam.read()
init_crop = init_frame[367:937, 536:1136,:]
# update boundary functions (assumes we have some blocks visible)
l_red, u_red = boundary_estimate(init_crop, lr_bound, ur_bound, 80, 255, 50, 220, 10)
l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound, 60, 255, 0, 255, 25)

running = 1
fcount = 1

print("Ready to record ")
while running:
    frame = picam.read()
    omni_frame = frame[367:967, 536:1136,:]

    # filter the image with these limit
    homing_frame = NavImage(omni_frame.copy())
    homing_frame.convertHsv()

    homing_frame.hsvMask(l_red, u_red)
    homing_frame.frame[wide_mask < 1] = 0

    green_frame = NavImage(omni_frame.copy())
    green_frame.convertHsv()
    green_frame.hsvMask(l_green, u_green)
    green_frame.frame[wide_mask < 1] = 0

    #_, cnts, _ = cv2.findContours(homing_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts_lg = [c for c in cnts if cv2.contourArea(c)>200]

    #for c in cnts_lg:
    #    print(cv2.contourArea(c))

    # Store
    #key = cv2.waitKey(1) & 0xFF
    #cv2.imshow("Visualisation", homing_frame.frame)

    if fcount%10 < 1:  #key == ord("a"):
        imgsave(omni_frame, './TestIm/SIFTtuning_', fcount)
        imgsave(homing_frame.frame, './TestIm/homeRed_', fcount)
        imgsave(green_frame.frame, './TestIm/homeGreen_', fcount)

    fcount += 1

    #elif key == ord("q"):
    #	running = False
    # break



# then return the size of the red contours

