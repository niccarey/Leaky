#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

erode_kernel = np.ones((7,7), np.uint8)
dilate_kernel = no.ones((7,7), np.uint8)

dilate_kernel_big = np.ones((11,11), np.uint8)
temp_blur_size = 11

# Define omnicam masks:
# ----------------------------------------------------------------------
cp = [ 300, 300 ]
r_out = 296;
r_inner = 150;
r_norim = 294;

y_crop_min = 293
y_crop_max = 893

x_crop_min = 594
x_crop_max = 1194

poly_front = np.array([cp, [1, 20], [1,1], [600,1], [600,20]])
poly_back = np.array([cp, [1, 430],[1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 430]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])

# ----------------------------------------------------------------------
# Define filter boundaries

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