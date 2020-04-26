#encoding: utf-8
# filename: speed_tests.py

import numpy as np
import cv2
from PIL import Image
import leaky_nav_speed as lns
import time
from KeypointExpanded import KeypointExpanded

cp = np.array([300,300])
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

l_red = np.array([0, 80, 50])
u_red = np.array([20, 255, 255])


xmap_ds, ymap_ds = lns.buildMap(300,300, 360, 180, 150, cp[0]/2, cp[1]/2)
xmap, ymap = lns.buildMap(600,600, 720, 360, 300, cp[0], cp[1])
sides_mask, front_mask, wide_mask = lns.define_masks([600, 600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)
probmat = np.zeros((6,6))
depvec = np.array([1.0, 1.0, 1.0, 1.0, 0, 0])

# most nav functions vastly sped up by installing openmp and some basic cythonizing
# BUT can't speed up sift and detect??
# Use SURF - it's optimized, performance is adequate (slightly less good than SIFT but not so much it probably amtters)

def nav_func_test(im1name, im2name):
    testim = cv2.imread(im1name)
    testcomp = cv2.imread(im2name)
    testim = cv2.cvtColor(testim, cv2.COLOR_RGB2BGR)
    testcomp = cv2.cvtColor(testcomp, cv2.COLOR_RGB2BGR)

    o_width, tracking_mask, unwrap_gray = lns.init_tracking_mask(xmap, ymap, l_red, u_red, testim, wide_mask)
    surf = cv2.xfeatures2d.SURF_create()
    kp_sift, des_sift = surf.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

    # something stupid happening with the masks here
    c_width, delta, h_store, tracking_comp, unwrap_gray_comp, home_check, red_cent = lns.run_tracking_mask(xmap, ymap, l_red, u_red, testcomp, wide_mask, o_width)
    InitialKeypoints = KeypointExpanded(kp_sift)
    a0 = -188.44
    a2 = 0.0072
    a3 = -0.0000374
    a4 = 0.0000000887
    xS = 125
    InitialKeypoints = lns.keypoint_height_calc(InitialKeypoints, a0, a2, a3, a4, xS)

    #storeIm = Image.fromarray(testcomp)
    #imname = 'check_second_im.jpg'
    #storeIm.save(imname)
    
    kp_comp_sift, des_comp_sift = surf.detectAndCompute(unwrap_gray_comp, tracking_comp.astype(np.uint8))
    imdisp = cv2.drawKeypoints(unwrap_gray_comp, kp_comp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    if (not (des_comp_sift is None)) :
        sift_matches = bf.match(des_sift, des_comp_sift) 
        sift_matches = sorted(sift_matches, key= lambda x:x.distance)
                    
        rotation, x_est, y_est = lns.est_egomotion(sift_matches, InitialKeypoints, kp_comp_sift)
        print(rotation, x_est, y_est)
