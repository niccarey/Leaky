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
picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()

time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.2)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain

time.sleep(0.5)

init_frame = picam.read()
init_crop = init_frame[360:960, 532:1132,:]

hist_frame = NavImage(init_crop)
hist_frame.convertHsv()
hist = cv2.calcHist([hist_frame.frame], [0], None, [180], [0, 180])
print(hist)
print(np.argmax(hist))
histmax = np.argmax(hist)


#plt.plot(hist)
#plt.xlim([0,180])
#plt.show()
#b,g,r = cv2.split(init_crop)
#new_im = cv2.merge([r,g,b])

l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound)

l_red = np.array([histmax - 20, 40,50])
u_red = np.array([histmax+20, 255, 255])

running = True

fcount = 1

while running:
    frame = picam.read()
    omni_frame = frame[360:960, 532:1132, :]
    #heading_angle, show_frame = omni_balance(cp, omni_frame, sides_mask, l_green, u_green)
    dcare1, dcare2, show_frame = omni_deposit(cp, omni_frame,wide_mask, l_green, u_green) 
    #red_locs, heading_angle, red_sizes, red_frame = omni_home(cp, omni_frame, wide_mask, l_red, u_red)
    #redImage = NavImage(omni_frame)
    #redImage.convertHsv()
    #redImage.hsvMask(l_red, u_red)
    #cv2.imshow('Checking output', show_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        running = False
        break

    img = Image.fromarray(show_frame)
    imname = './TestIm/GreenOutput_'
    imname += str(fcount)
    imname += '.png'
    img.save(imname)

    fcount += 1

#hist_short = hist[lg_bound:ug_bound]
#plt.plot(hist_short)
#plt.xlim([0,110])
#plt.show()
#print(np.argmax(hist_short))


#red_peak = lg_bound + np.argmax(hist_short)
#l_red = np.array([red_peak - 20, 100, 30])
#u_red = np.array([red_peak + 20, 255, 255])

#time.sleep(0.2)

#hist_frame.hsvMask(l_red, u_red)
#imgmask = Image.fromarray(hist_frame.frame)
#immaskname = './TestIm/maskoutput.png'
#imgmask.save(immaskname)

#time.sleep(2)
