#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Necessary navigation functions

import cv2
from navImage import navImage
import numpy as np 
import multiprocessing as mp 
import kernel_def
import RPi.GPIO as gp
import time

# PROBABILITY CONSTANTS
p_bend_LS = 0.81
p_whratio_LS = 0.56
p_nwalls_LS = 0.51

p_set_LAv = 0.04


# MASKS

def define_masks():
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

def get_striped_mask(o_width, ymin, ymax, xmin, xmax, unwrapshape):
       
    stripe_mask_corners = np.array([[xmin, ymax], [xmin, ymin], [xmax,ymin], [xmax,ymax]])
    mask_shape = unwrapshape
    stripe_mask = np.zeros((mask_shape))
    cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

    return stripe_mask

def init_tracking_mask(xmap, ymap, l_red, u_red, col_frame, wide_mask):    
    gray_frame = cv2.cvtColor(col_frame, cv2.COLOR_BGR2GRAY)
    unwrap_gray = unwarp(gray_frame, xmap, ymap)
    unwrap_init = unwarp(col_frame, xmap, ymap)
    
    wide_mask[gray_frame>230] = 0
    wide_erode = cv2.erode(wide_mask.astype(np.uint8), erode_kernel, iterations=1)
    mask_unwrap = unwarp(wide_erode, xmap, ymap)
    
    home_frame = get_home_frame(l_red, u_red, unwrap_init)    
    
    _, cnts, _ = cv2.findContours(home_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>300]
    xmin = 720
    xmax = 0
    for c in cnts_lg:
        x, y, w, h = cv2.boundingRect(c)
        if x > 100 and x < 550:
            if x+w > xmax: xmax = x+w
            if x < xmin: xmin = x

    owidth = xmax - xmin 
    stripe_mask = get_striped_mask(owidth, 180,360, xmin, xmax, mask_unwrap.shape)
    
    tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
    tracking_mask[tracking_mask>0] = 255

    return owidth, tracking_mask, unwrap_gray


# FILTERS

def boundary_estimate(frame, l_bound, u_bound, l_sat, u_sat, l_val, u_val, dh):
    # convert to HSV
    hist_frame = NavImage(frame)
    hist_frame.convertHsv()

    hist = cv2.calcHist([hist_frame.frame], [0], None, [180], [0, 180])
    #print(hist)
    hist_short = hist[l_bound:u_bound]

    col_peak = l_bound + np.argmax(hist_short)
    print(col_peak)
    l_return = np.array([col_peak - dh, l_sat, l_val])
    u_return = np.array([col_peak + dh, u_sat, u_val])

    return l_return, u_return

def get_home_frame(l_red, u_red, col_frame):
    
    redbar_frame = NavImage(col_frame)
    redbar_frame.convertHsv()
    redbar_frame.hsvMask(l_red, u_red)
    redbar_frame.erodeMask(erode_kernel, 1)
    redbar_frame.dilateMask(dilate_kernel_big, 1)

    return(redbar_frame.frame)

    
def lpfilter(input_buffer):
    # check length
    if len(input_buffer) < 3:
        return 0
    
    else:
        output = 0.6*input_buffer.pop() + 0.2*input_buffer.pop() + 0.2*input_buffer.pop()
        return output



# MAPS

def buildMap(widthS, heightS, widthD, heightD, radS, cx, cy):
    # set up destination maps for x, y coordinates
    map_x = np.zeros((heightD, widthD), np.float32)
    map_y = np.zeros((heightD, widthD), np.float32)
    for y in range(0, int(heightD-1)):
        for x in range(0, int(widthD-1)):
            # work out azimuth, elevation based on DEST image (rho)
            rho = (float(y)/float(heightD))*radS
            theta = (float(x)/float(widthD))*2.0*np.pi # just converts degree to radians
            # corresponding positions in SOURCE image:
            xS = cx + rho*np.sin(theta)
            yS = cy + rho*np.cos(theta)
            map_x.itemset((y,x), int(xS))
            map_y.itemset((y,x), int(yS))            
    
    return map_x, map_y

def unwarp(img_array, xmap, ymap):
	output = cv2.remap(img_array, xmap, ymap, cv2.INTER_LINEAR)
	return output

# IMAGE PROCESSING

def omni_balance():
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


def omni_deposit():
    dep_frame = NavImage(omni_frame)
    dep_frame.convertHsv()
    dep_frame.hsvMask(l_green, u_green)
    dep_frame.frame[mask < 1] = 0
        
    # turn until we have one ROI of a sufficient size, with CoM within a central window
    dep_frame.dilateMask(dilate_kernel, 1)
    dep_frame.erodeMask(erode_kernel, 1)
    dep_frame.dilateMask(dilate_kernel_big, 1)

    # problem is warp
    dep_frame_unwarp = unwarp(dep_frame.frame.copy(), xmap, ymap)
    _, cnts, _ = cv2.findContours(dep_frame_unwarp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # check how many segments larger than (threshold)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>1000]

    if len(cnts_lg) > 0: 
        rect = cv2.minAreaRect(cnts_lg[0])
        box= np.int0(cv2.boxPoints(rect))
        maxbox = max(box[:,0])
        minbox = min(box[:,0])
        if (rect[2] > -45) and (rect[2] < 45):
            boxratio = rect[1][0]
            boxratio /= rect[1][1]
        else:
            boxratio = rect[1][1]
            boxratio /= rect[1][0]
        #else: # Ideally we would turn until wall and blob were merged, but this is hard to generalise
           
        M = cv2.moments(cnts_lg[0])                
        cy = int(M['m10']/M['m00'])
        heading_angle = (float(cy)/2)*np.pi/180

        return len(cnts_lg), heading_angle, np.array([minbox, maxbox]), boxratio, dep_frame_unwarp

    else:
        return 0, 0, 0, 0, dep_frame.frame.copy()

def omni_home():
    back_frame = NavImage(omni_frame)
    back_frame.convertHsv()
    back_frame.hsvMask(l_red, u_red)
    back_frame.frame[mask<1] = 0

    back_frame.erodeMask(erode_kernel, 1)
    back_frame.dilateMask(dilate_kernel,1)
    _, cnts, _ = cv2.findContours(back_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnts_lg = [c for c in cnts if cv2.contourArea(c)> 100]

        xmin = 600
	    xmax = 0
	    ymin = 0
   		hstore = 300
    	for c in cnts_lg:
        	x, y, w, h = cv2.boundingRect(c)
        	if y > ymin: 
            	ymin = y
            	hstore = h
        	if (x > 50) and (x< 650):
            	if x+w > xmax: xmax = x+w
            	if x < xmin : xmin = x

	    cwidth = xmax - xmin
	    centre_approx = (xmax + xmin)/2

        if len(cnts_lg) > 0:
            return len(cnts_lg), centre_approx, back_frame.frame.copy()

        else: return 0, 0, back_frame.frame.copy()

    else: return 0,0 , back_frame.frame.copy()

def homing_direction(red_cent):
	if red_cent < 340:
		leaky.cam_flag = 1
		leaky.direction = 'left'

	elif red_cent > 380:
		leaky.cam_flag = 1
		leaky.direction = 'right'

	else:
		leaky.direction = 'fwd'


def leaving_home():
    leave_frame = NavImage(omni_frame)
    leave_frame.convertHsv()
    leave_frame.hsvMask(l_red, u_red)
    leave_frame.frame[wide_mask < 1] = 0
    leave_frame.erodeMask(erode_kernel, 1)
    leave_frame.dilateMask(dilate_kernel,1)

    _, cnts, _ = cv2.findContours(leave_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)> 900]
    cnts_sorted = sorted(cnts_lg, key=cv2.contourArea, reverse=True)
    if len(cnts_lg)>1:
        sumx = 0
        for i in range(0,1):
            M = cv2.moments(cnts_sorted[i])
            cx = int(M['m01']/M['m00'])-cp[1]
            print(cx)
            sumx += cx

    elif len(cnts_lg) == 1:
        M =cv2.moments(cnts_sorted[0])
        cx = int(M['m01']/M['m00']) - cp[1]

    else: cx = cp[1]

    return len(cnts_lg), (cx - cp[1]), leave_frame.frame.copy()

# LOCALISATION

def localisation_prob(prev_prob, bend_exist, ratio_exist, walls_exist):

	# Calculate new localisation probability based on observations:
	if bend_exist: p_bend = p_bend_LS
	else: p_bend = 1-p_bend_LS

	if ratio_exist: p_ratio = p_whratio_LS
	else: p_ratio = 1 - p_whratio_LS

	if walls_exist: p_walls = p_nwalls_LS
	p_walls = 1 - p_nwalls_LS

	p_local_new = p_bend*p_ratio*p_walls
    p_normalise = p_local_new*prev_prob + (1-prev_prob)*p_set_LAv

	p_update = p_local_new*prev_prob/p_normalise

	return p_update


# TOUCH AND DISTANCE SENSING

def get_block_reading(reflectorPin):

    gp.setup(reflectorPin, gp.OUT)
    gp.output(reflectorPin, True)
    time.sleep(0.02)

    count = 0
    gp.setup(reflectorPin, gp.IN)
    while gp.input(reflectorPin) == True:
    	count = count +1

    block_reading = count/10

	return block_reading


def get_adc_reading(adc, sensor_channel, gain):

    value = adc.read_adc(sensor_channel, gain=gain)
    return value


