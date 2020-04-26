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
    imname += '.jpg'
    storeIm.save(imname)


# Define omnicam masks:
cp = [ 300, 300 ]
r_out = 298;
r_inner = 150;
r_norim = 295;

poly_front = np.array([cp, [20, 1], [600,1]])
poly_back = np.array([cp, [1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])

sides_mask, front_mask, wide_mask = define_masks([600,600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

lg_bound = 40
ug_bound = 80

lr_bound = 90
ur_bound = 120

picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()

time.sleep(0.5)

print(" ... setting up unwarp map ...")
xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])
print("...done")

# To start:
# we want to tune the red filter like the green filter

init_frame = picam.read()
init_crop = init_frame[367:937, 536:1136,:]
# update boundary functions (assumes we have some blocks visible)
l_red, u_red = boundary_estimate(init_crop, lr_bound, ur_bound, 80, 255, 50, 220, 15)
l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound, 50, 255, 0, 255, 25)

running = 1
fcount = 1

print("Ready to record ")
while running:
    frame = picam.read()
    omni_frame = frame[320:920, 530:1130,:]

    # filter the image with these limit
    #homing_frame = NavImage(omni_frame.copy())
    #homing_frame.convertHsv()

    #homing_frame.hsvMask(l_red, u_red)
    #homing_frame.frame[wide_mask < 1] = 0

    green_frame = NavImage(omni_frame.copy())
    green_frame.convertHsv()
    green_frame.hsvMask(l_green, u_green)
    green_frame.frame[wide_mask < 1] = 0
    dep_frame_unwarp = unwarp(green_frame.frame.copy(), xmap, ymap)

    unwarp_col = unwarp(omni_frame, xmap, ymap)
 
    _, cnts, _ = cv2.findContours(dep_frame_unwarp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>200]
    if len(cnts_lg) > 0:
        rect = cv2.minAreaRect(cnts_lg[0])
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(unwarp_col, [box], 0, (0,255,0), 2)
        if (rect[2] > -45) and (rect[2] < 45):
            boxratio = rect[1][0]
            boxratio /= rect[1][1]
        else:
            boxratio = rect[1][1]
            boxratio /= rect[1][0]
            
        print(max(box[:,0]) - min(box[:,0]))
        M = cv2.moments(cnts_lg[0])     
        cy = int(M['m10']/M['m00']) #- cp[0]
        heading_angle = float(cy)/2
        #print(heading_angle)

    # Store
    key = cv2.waitKey(1) & 0xFF
    #cv2.imshow("Visualisation", unwarp_col)

    if fcount%10 < 1:  #key == ord("a"):
    #    imgsave(unwarp_col, './TestIm/Turn2Dep_test__', fcount)
        imgsave(omni_frame, './TestIm/testingCrop_', fcount)
        #imgsave(dep_frame_unwarp, './TestIm/homeGreen_', fcount)

    fcount += 1

    if key == ord("q"):
    	running = False
        break



# then return the size of the red contours

