#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from navImage import NavImage
import numpy as np

erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)
temp_blur_size = 11

# define masks
def define_masks(imshape, cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right):
    xpix, ypix = imshape
    print(xpix, ypix)
    y,x = np.ogrid[0:ypix, 0:xpix]

    outer_mask = np.zeros((ypix,xpix))
    omask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_out**2
    outer_mask[omask_px] = 1


    inner_mask = np.zeros((ypix,xpix))
    imask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_inner**2
    inner_mask[imask_px] = 1

    rim_mask = np.zeros((ypix,xpix))
    rmask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_norim**2
    rim_mask[rmask_px] = 1

    omni_ring = outer_mask - inner_mask
    omni_thin = rim_mask - inner_mask


    # define polygons for ROI
    front_mask = np.zeros((ypix, xpix))
    cv2.fillConvexPoly(front_mask, poly_front, 1)

    # back and sides
    back_mask = np.zeros((ypix,xpix))
    cv2.fillConvexPoly(back_mask, poly_back, 1)

    left_mask = np.zeros((ypix,xpix))
    cv2.fillConvexPoly(left_mask, poly_left, 1)

    right_mask = np.zeros((ypix,xpix))
    cv2.fillConvexPoly(right_mask, poly_right, 1)

    # Define masks
    fb_region = front_mask+back_mask
    sides_mask = omni_ring - fb_region
    sides_mask[sides_mask<0] = 0

    sb_region = back_mask + left_mask + right_mask
    front_mask = omni_ring - sb_region
    front_mask[front_mask<0] = 0

    wide_mask = omni_thin - back_mask
    wide_mask[wide_mask<0] = 0

    return sides_mask, front_mask, wide_mask


def boundary_estimate(frame, lg_bound, ug_bound):
    # convert to HSV
    hist_frame = NavImage(frame)
    hist_frame.convertHsv()
    
    hist = cv2.calcHist([hist_frame.frame], [0], None, [180], [0, 180])
    hist_short = hist[lg_bound:ug_bound]
    
    green_peak = lg_bound + np.argmax(hist_short)
    print(green_peak)
    l_green = np.array([green_peak - 25, 20, 0])
    u_green = np.array([green_peak + 25, 255, 255])
    
    return l_green, u_green
    

def probability_calculator(numblob, whratio):
	retvec = np.array([0 0 ])
	if numblob > 2: retvec[0] = 0.1
	elif numblob > 1: retvec[0] = 0.2
	else: retvec[0] = 0.3
	
	if whratio > 3: retvec[1] = 0.3
	elif whratio > 2.8: retvec[1] = 0.2
	else: retvec[1] = 0.1
	
	return retvec
	
def localisation_calculator(locvec, probmatrix):
    probvec = np.dot(probmatrix, locvec)
    probscalar = np.sum(probvec)

	return probscalar
 

def omni_balance(cp, omni_frame, mask, l_green, u_green):
    # apply mask
    bal_frame = NavImage(omni_frame)
    bal_frame.convertHsv()
    bal_frame.hsvMask(l_green, u_green)
    
    bal_frame.frame[mask < 1] = 0
        
    bal_frame.erodeMask(erode_kernel, 1)
    bal_frame.dilateMask(dilate_kernel, 1)
        
    _, cnts, _ = cv2.findContours(bal_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if len(cnts)>0:
        # find contours with area > (some threshold) (there is a more elegant way of doing this)
        cont_max = max(cnts, key = cv2.contourArea)
        # y is vertical, x is horizontal
        sum_w_x = 0
        sum_w_y = 0
        cnts_lg = [c for c in cnts if cv2.contourArea(c)>2000]
            
        for cnt in cnts_lg:
            M = cv2.moments(cnt)
            cent_ang = np.arctan2(int(M['m10']/M['m00']) - cp[0],int(M['m01']/M['m00']) - cp[1])
            blob_area = cv2.contourArea(cnt)

            dirvec_w = np.array((np.sin(-cent_ang), np.cos(-cent_ang)),dtype = np.float)*blob_area/cv2.contourArea(cont_max)
                
            sum_w_y += dirvec_w[0]
            sum_w_x += dirvec_w[1]
            

        heading_angle = np.arctan2(sum_w_y, sum_w_x)
        if (0 < heading_angle < np.pi/2):
            heading_angle = heading_angle+np.pi
        elif (-np.pi/2 < heading_angle < 0):
            heading_angle = heading_angle - np.pi

        return heading_angle, bal_frame.frame.copy()

    else:
        return 0, bal_frame.frame.copy()



def omni_deposit(cp, omni_frame, mask, l_green, u_green):
    dep_frame = NavImage(omni_frame)
    dep_frame.convertHsv()
    dep_frame.hsvMask(l_green, u_green)
    dep_frame.frame[mask < 1] = 0
        
    # turn until we have one ROI of a sufficient size, with CoM within a central window
    dep_frame.dilateMask(dilate_kernel, 1)
    dep_frame.erodeMask(erode_kernel, 1)
    dep_frame.dilateMask(dilate_kernel, 1)

    _, cnts, _ = cv2.findContours(dep_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # check how many segments larger than (threshold)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>800]

    if len(cnts_lg) > 0:
        rect = cv2.minAreaRect(cnts_lg[0])
        if (rect[2] > -45) and (rect[2] < 45):
            boxratio = rect[1][0]
            boxratio /= rect[1][1]
        else:
            boxratio = rect[1][1]
            boxratio /= rect[1][0]
        #print("w/h ratio: ", boxratio, "angle: ", rect[2])
        
        #if (len(cnts_lg) > 1) or (boxratio < 2.52) :
        #    #print("Keep turning, can see: ", len(cnts_lg))
        #    return len(cnts_lg), 0, boxratio, dep_frame.frame.copy()
        
        #else: # Ideally we would turn until wall and blob were merged, but this is hard to generalise
        blob = cnts_lg[0]
            
        M = cv2.moments(blob)                
        cy = int(M['m10']/M['m00']) - cp[0]
        cx = int(M['m01']/M['m00']) - cp[1]
        heading_angle = np.arctan2(cy,cx)

        return len(cnts_lg), heading_angle, boxratio, dep_frame.frame.copy()

    else:
        return 0, 0, 0, dep_frame.frame.copy()


def omni_home(cp, omni_frame, mask, l_red, u_red):
    back_frame = NavImage(omni_frame)
    back_frame.convertHsv()
    back_frame.hsvMask(l_red, u_red)
    back_frame.frame[mask < 1] = 0
        
    # turn until we can see two contours in roughly the right position
    back_frame.erodeMask(erode_kernel,1)
    back_frame.dilateMask(dilate_kernel, 1)
        
    _, cnts, _ = cv2.findContours(back_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if len(cnts) > 0:
        cnts_lg = [c for c in cnts if cv2.contourArea(c)>100]
   
        if len(cnts_lg)> 0 :
            sum_x = 0
            sum_y = 0
            cnt_sort = sorted(cnts_lg, key=cv2.contourArea, reverse=True)

            for c in cnt_sort:
                M = cv2.moments(c)

                cy = int(M['m10']/M['m00']) - cp[0]
                cx = int(M['m01']/M['m00']) - cp[1]
                cent_ang = np.arctan2(cy,cx)
                
                dirvec = [np.sin(cent_ang), np.cos(cent_ang)]
                sum_y += dirvec[0]
                sum_x += dirvec[1]
                
            heading_angle = np.arctan2(sum_y, sum_x)
            c_area_list = [cv2.contourArea(c) for c in cnt_sort]

            return len(cnts_lg), heading_angle, c_area_list, back_frame.frame.copy()

        else: 
            return 0,0,0,back_frame.frame.copy()

    else:
        return 0, 0, 0, back_frame.frame.copy()


def sim_mse(imA, imB):
    err = np.sum((imA.astype("float") - imB.astype("float"))**2)
    err /= float(imA.shape[0] * imA.shape[1])
    return err

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
    

