import numpy as np
import cv2
from PIL import Image
import leaky_nav_functions as lnf
import time

cp = [300,300]

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

l_green = np.array([40, 60, 0])
u_green = np.array([90, 255, 255])


#def nav_func_test(imname):
sides_mask, front_mask, wide_mask = lnf.define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)
testim = cv2.imread('./SpeedTesting/testspeed_im.png')
start = time.time()
heading_angle, show_frame = lnf.omni_balance(cp, testim, sides_mask, l_green, u_green)
print(time.time()-start)
