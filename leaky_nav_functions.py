#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from navImage import NavImage
import numpy as np

erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)
temp_blur_size = 11

def balance_detection(left_frame, right_frame, lb_left, lb_right, ub_left, ub_right, lscale, rscale, l_offset, r_offset):
    leftIm = NavImage(left_frame)
    rightIm = NavImage(right_frame)
    
    leftIm.convertHsv()
    rightIm.convertHsv()
    
    
    leftIm.hsvMask(lb_left, ub_left)
    rightIm.hsvMask(lb_right, ub_right)
    
    leftIm.erodeMask(erode_kernel, 1)
    rightIm.erodeMask(erode_kernel,1)
    rightIm.dilateMask(dilate_kernel, 1)
    
    left_weight = leftIm.maskWeight(lscale, l_offset)
    right_weight = rightIm.maskWeight(rscale, r_offset)
    
    weight_diff = (left_weight - right_weight)
    
    return (weight_diff, left_weight, right_weight)

def find_centroid(nav_frame, l_bound, u_bound, thresh_state, cam_flag, location):
    global cX, cY
    
    nav_image = NavImage(nav_frame) 
    nav_image.convertHsv()     
    nav_image.hsvMask(l_bound, u_bound)
    
    nav_image.erodeMask(erode_kernel, 1) 
    nav_image.dilateMask(dilate_kernel, 1)
    blur_im = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)
    mments = cv2.moments(blur_im)

    
    if (mments['m00'] > 0):
        cX = int(mments['m10']/mments['m00'])
        cY = int(mments['m01']/mments['m00'])
        
        if cam_flag:
            print("left centroid: ", cX)
        else:
            print("right centroid: " , cX)

        if thresh_state == 'leq' and (cX < location):
            #print("Passed threshold. leq")
            return True
        
        elif thresh_state == 'geq' and (cX > location):
            #print("Passed threshold, geq")
            return True  
            
        else:
            return False
    
    elif np.count_nonzero(blur_im) > 200000: # most of the screen is block
        print("uhoh, too close!")
        return True
        
    else:
        return False       


def find_edge(nav_frame, l_bound, u_bound, thresh_state, cam_flag, location):
    #global cX, cY
    
    nav_image = NavImage(nav_frame) 
    nav_image.convertHsv()     
    nav_image.hsvMask(l_bound, u_bound)
    
    
    # additional operations: probably don't need
    nav_image.erodeMask(erode_kernel, 1) 
    nav_image.dilateMask(dilate_kernel, 1)
    blur_im = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)

    
    if cam_flag: #(direction == 'left'):
        template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), left_template, cv2.TM_CCORR_NORMED)
        
    else:
        template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), right_template, cv2.TM_CCORR_NORMED)
    
    # Locate best template match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    #print max_val

    if (max_val > 0.7) and (max_loc[0]>2):
        # we are fairly sure we have found a good match
        # get true edge location
        print("Looking for: ", thresh_state, location)
        cX = max_loc[0]+temp_size//2 
        cY = max_loc[1] 
        #print(cX, max_val)
        
        if thresh_state == 'leq' and (max_loc[0] + temp_size//2 < location):
            #print("Passed threshold. leq")
            return True
        
        elif thresh_state == 'geq' and (max_loc[0] + temp_size//2 > location):
            #print("Passed threshold, geq")
            return True  
            
        else:
            return False
    
    elif np.count_nonzero(blur_im) > 200000: # most of the screen is block
        print("uhoh, too close!")
        return True
        
    else:
        return False    

    
def lpfilter(input_buffer):
    # check length
    if len(input_buffer) < 3:
        return 0
    
    else:
        output = 0.6*input_buffer.pop() + 0.2*input_buffer.pop() + 0.2*input_buffer.pop()
        return output


# Should'nt need this now
def check_edges(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, found_edge_threshold, back_up_threshold):
    global deposit_edge
    global backup_edge
    
    deposit_edge = find_edge(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, found_edge_threshold)
    backup_edge = find_edge(nav_frame, prev_frame, l_bound, u_bound, threshold_state, cam_flag, back_up_threshold)
    if deposit_edge:
        print("deposit: TRUE")
        
    if backup_edge:
        print("Backup: TRUE")


def check_balance(lframe, rframe, ll_bound, ul_bound, lr_bound, ur_bound, left_scale, right_scale, left_offset, right_offset, wall_flag): #ubl, ubr, bl_scale, br_scale, lb_offset, rb_offset, lrl, lrr, url, urr, rl_scale, rr_scale, lr_offset, rr_offset):
    global blue_walls_match
    global home_match
    
    global left_home, right_home
    global blue_left_wall, blue_right_wall
    
    if wall_flag: 
        blue_walls_match, blue_left_wall, blue_right_wall = balance_detection(lframe, rframe, ll_bound, lr_bound, ul_bound, ur_bound, left_scale, right_scale, left_offset, right_offset)
        
    else: 
        home_match, left_home, right_home  = balance_detection(lframe, rframe, ll_bound, lr_bound, ul_bound, ur_bound, left_scale, right_scale, left_offset, right_offset)
    
    
