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
cp = [ 389, 297]
r_out = 290;
r_inner = 145;
r_norim = 260;

poly_front = np.array([cp, [20, 1], [620,1]])
poly_back = np.array([cp, [1, 600], [800,600], [800,420]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [620, 1], [800 ,1], [800, 420]])

sides_mask, front_mask, wide_mask = define_masks(cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)

lg_bound = 150
ug_bound = 180
picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()

time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.2)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain

time.sleep(0.3)

init_frame = picam.read()
init_crop = init_frame[196:796, 432:1232,:]
b,g,r = cv2.split(init_crop)
new_im = cv2.merge([r,g,b])
#l_green, u_green = boundary_estimate(init_crop, lg_bound, ug_bound)

hist_frame = NavImage(init_crop)
hist_frame.convertHsv()

hist = cv2.calcHist([hist_frame.frame], [0], None, [180], [lg_bound, ug_bound])

plt.plot(hist)
plt.xlim([0,180])
plt.show()

img = Image.fromarray(new_im)
imname = './TestIm/Histoutput.png'
img.save(imname)

hist_short = hist[lg_bound:ug_bound]
plt.plot(hist_short)
plt.xlim([0,110])
plt.show()
print(np.argmax(hist_short))


red_peak = lg_bound + np.argmax(hist_short)
l_red = np.array([red_peak - 20, 100, 30])
u_red = np.array([red_peak + 20, 255, 255])

time.sleep(0.2)

hist_frame.hsvMask(l_red, u_red)
imgmask = Image.fromarray(hist_frame.frame)
immaskname = './TestIm/maskoutput.png'
imgmask.save(immaskname)

time.sleep(2)
