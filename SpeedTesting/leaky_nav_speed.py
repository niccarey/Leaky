#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from navImage import NavImage
import numpy as np
from PIL import Image # can remove this after debug
import multiprocessing as mp

erode_kernel = np.ones((7,7), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)

dilate_kernel_big = np.ones((11,11), np.uint8)
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
    
def get_home_frame(l_red, u_red, col_frame):
    
    redbar_frame = NavImage(col_frame)
    redbar_frame.convertHsv()
    redbar_frame.hsvMask(l_red, u_red)
    redbar_frame.erodeMask(erode_kernel, 1)
    redbar_frame.dilateMask(dilate_kernel_big, 1)

    return(redbar_frame.frame)
    
def get_striped_mask(o_width, ymax, ymin, xmin, xmax, unwrapshape):
       
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


def run_tracking_mask( xmap, ymap, l_red, u_red, col_frame, wide_mask, owidth):
    gray_frame = cv2.cvtColor(col_frame, cv2.COLOR_BGR2GRAY)
    unwrap_gray = unwarp(gray_frame, xmap, ymap)
    unwrap_col = unwarp(col_frame, xmap, ymap)    

    home_frame = get_home_frame(l_red, u_red, unwrap_col)

    wide_mask[gray_frame>230] = 0
    wide_erode = cv2.erode(wide_mask.astype(np.uint8), erode_kernel, iterations=1)
    mask_unwrap = unwarp(wide_erode, xmap, ymap)

    _, cnts, _ = cv2.findContours(home_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>350]
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
    delta = float(cwidth)/float(owidth)

    if delta < 1:
        smask_y = int(delta*180)
    else:
        smask_y = 180

    stripe_mask = get_striped_mask(cwidth, smask_y, ymin, xmin, xmax, mask_unwrap.shape)

    tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
    tracking_mask[tracking_mask>0] = 255
    
    return cwidth, delta, hstore, tracking_mask, unwrap_gray, home_frame, centre_approx
    
def keypoint_height_calc(IK, a0, a2, a3, a4, xS):    
    for kp_it in IK.keypoints:
        uc_w = (kp_it.pt[0])*np.pi/(4*180)
        vc_w = kp_it.pt[1]*300/360
        # Convert to omnicam pixel positions
        uc =  vc_w*np.sin(uc_w)
        vc =  vc_w*np.cos(uc_w)
        #print(uc,vc)
        rho = np.sqrt(np.square(uc) + np.square(vc ))
        frho = a0 + a2*np.square(rho) + a3*np.power(rho,3) + a4*np.power(rho,4)
        lambda_est = xS/(uc)
        # invert because calibration assumes an upside-down mirror
        z_est = -(lambda_est*frho)
        IK.add_height([z_est])
        
    return IK
    
        
def mask_weight_calcs(cwidth, owidth):
    delta = float(cwidth)/float(owidth)
    if delta < 1:
        ratio_weight = 0
        smask_y = int(delta*180)
    else: 
        smask_y = 180
        ratio_weight = 0.2

    if hstore > 84: height_weight = 0.3
    else: height_weight = 0
    
    return ratio_weight, height_weight
    
def est_egomotion(sift_matches, InitialKeypoints, kp_comp_sift):    
    Alist = []
    Blist = []
    for siftdat in sift_matches[:15]:
        # retrieve data:
        kp_retrieve = InitialKeypoints.keypoints
        height_retrieve = InitialKeypoints.heights
            
        # calculate omnicam pixel positions using kp_retrieve
        old_uc_w = (kp_retrieve[siftdat.queryIdx].pt[0])*np.pi/(4*180)
        old_vc_w = (kp_retrieve[siftdat.queryIdx].pt[1])*300/360
        old_uc =  old_vc_w*np.sin(old_uc_w) 
        old_vc =  old_vc_w*np.cos(old_uc_w) 
        h_pt = height_retrieve[siftdat.queryIdx]
            
        new_uc_w = (kp_comp_sift[siftdat.trainIdx].pt[0])*np.pi/(4*180)
        new_vc_w = (kp_comp_sift[siftdat.trainIdx].pt[1])*300/360
        new_uc =  new_vc_w*np.sin(new_uc_w) 
        new_vc =  new_vc_w*np.cos(new_uc_w) 
            
        # create point-based matrices and append to existing structure 
        Apoint = np.array([[old_uc, -old_vc, -1/h_pt, 0],[old_vc, old_uc, 0, -1/h_pt]])
        Bpoint = np.array([new_uc, new_vc])

        Alist.extend(Apoint,)
        Blist.extend(Bpoint,)
            
    Alist = np.array(Alist)
    Blist = np.array(Blist)
    s_vec = np.dot(np.linalg.pinv(Alist), np.transpose(Blist))
    Qmat = np.array([[s_vec[0], -s_vec[1]], [s_vec[1], s_vec[0]]])
    U, s, V = np.linalg.svd(Qmat)
    Rmat = np.dot(U, np.transpose(V))
    rotation = np.arctan(Rmat[1,0]/Rmat[0,0])
    transx = s_vec[2]
    transy = s_vec[3]
    
    return rotation, transx, transy
    

def dep_prob_calculator(numblob, whratio, minmax):

    if numblob > 2: blob_prob = 0.2
    elif numblob > 1: blob_prob = 0.3
    else: blob_prob = 0.4

    if whratio > 1.5 or (min(minmax)<250) or (max(minmax)>500) : width_prob = 0.3
    elif whratio > 1.3: width_prob = 0.2
    else: width_prob = 0.1

    retvec= np.array([blob_prob, width_prob])

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



def omni_deposit(cp, omni_frame, mask, l_green, u_green, xmap, ymap):
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

def omni_home(cp, omni_frame, mask, l_red, u_red):
    back_frame = NavImage(omni_frame)
    back_frame.convertHsv()
    back_frame.hsvMask(l_red, u_red)
    back_frame.frame[mask<1] = 0

    back_frame.erodeMask(erode_kernel, 1)
    back_frame.dilateMask(dilate_kernel,1)
    _, cnts, _ = cv2.findContours(back_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnts_lg = [c for c in cnts if cv2.contourArea(c)> 100]

        if len(cnts_lg) > 0:
            return len(cnts_lg), back_frame.frame.copy()

        else: return 0,back_frame.frame.copy()

    else: return 0,back_frame.frame.copy()

def leaving_home(cp, omni_frame, wide_mask, l_red, u_red):
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


def sim_mse(imA, imB):
    err = np.sum((imA.astype("float") - imB.astype("float"))**2)
    err /= float(imA.shape[0] * imA.shape[1])
    return err

    
def lpfilter(input_buffer):
    # check length
    if len(input_buffer) < 3:
        return 0
    
    else:
        output = 0.6*input_buffer.pop() + 0.2*input_buffer.pop() + 0.2*input_buffer.pop()
        return output


