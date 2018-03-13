#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from navImage import NavImage
import numpy as np
from PIL import Image # can remove this after debug
import multiprocessing as mp
import time

cv2.setNumThreads(4)

from cpython cimport array
import array
cimport numpy as np
DTYPE = np.int

ctypedef np.int_t DTYPE_t

cdef np.ndarray erode_kernel = np.ones((3,3), np.uint8)
cdef np.ndarray dilate_kernel = np.ones((3,3), np.uint8)

cdef np.ndarray dilate_kernel_big = np.ones((7,7), np.uint8)

# define masks
def define_masks(imshape, cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right):
    xpix, ypix = imshape
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


def convertMask(np.ndarray imconvert, np.ndarray lower_bound, np.ndarray upper_bound, np.ndarray apply_mask, int iteration, np.ndarray ek, np.ndarray dk):
    # instead of formal low-pass filter, we downsample and use modified indexes in outer function calls
    cdef np.ndarray init_mask, final_mask
    init_mask =cv2.inRange(cv2.cvtColor(imconvert[0:imconvert.shape[0]:2, 0:imconvert.shape[1]:2,:], cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
    init_mask[apply_mask[0:apply_mask.shape[0]:2, 0:apply_mask.shape[1]:2] < 1] = 0
    final_mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, ek)
    return final_mask
    
    
def convertMask_full(np.ndarray imconvert, np.ndarray lower_bound, np.ndarray upper_bound, np.ndarray apply_mask, int iteration, np.ndarray ek, np.ndarray dk):
    # instead of formal low-pass filter, we downsample and use modified indexes in outer function calls
    cdef np.ndarray init_mask, final_mask
    init_mask =cv2.inRange(cv2.cvtColor(imconvert, cv2.COLOR_BGR2HSV), lower_bound, upper_bound)
    init_mask[apply_mask < 1] = 0
    final_mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, ek)
    return final_mask

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
    
def unwarp(np.ndarray img_array, np.ndarray xmap, np.ndarray ymap):
    cdef np.ndarray output
    output = cv2.remap(img_array, xmap, ymap, cv2.INTER_LINEAR)
    return output
    
def get_home_frame(np.ndarray l_red, np.ndarray u_red, np.ndarray col_frame):
    cdef np.ndarray init_mask
    cdef np.ndarray redbar_frame
    
    init_mask =cv2.inRange(cv2.cvtColor(col_frame, cv2.COLOR_BGR2HSV), l_red, u_red)
    redbar_frame = cv2.dilate(cv2.erode(init_mask, erode_kernel, iterations=1), dilate_kernel_big, iterations=1)

    return(redbar_frame)

    
def get_striped_mask(int o_width, int ymax, int ymin, int xmin, int xmax, tuple unwrapshape):
    cdef np.ndarray stripe_mask_corners
    cdef np.ndarray stripe_mask
    
    stripe_mask_corners = np.array([[xmin, ymax], [xmin, ymin], [xmax,ymin], [xmax,ymax]])
    stripe_mask = np.zeros((unwrapshape))
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



def run_tracking_mask(np.ndarray xmap, np.ndarray ymap, np.ndarray l_red, np.ndarray u_red, np.ndarray col_frame, np.ndarray wide_mask, int owidth):
    # Slow: would like to downsample to increase speed BUT must make sure not to lose info
    
    cdef np.ndarray unwrap_gray
    cdef np.ndarray unwrap_col
    cdef np.ndarray home_frame
    cdef int xmin
    cdef int xmax
    cdef int ymin
    cdef int hstore
    cdef int x, y, w, h
    
    cdef np.ndarray tracking_mask
    cdef np.ndarray stripe_mask
    cdef int cwidth
    cdef double delta
    cdef int centre_approx
    
    gray_frame = cv2.cvtColor(col_frame, cv2.COLOR_BGR2GRAY)
    unwrap_gray = unwarp(gray_frame, xmap, ymap)
    unwrap_col = unwarp(col_frame, xmap, ymap)

    wide_mask[gray_frame>240] = 0
    mask_unwrap = unwarp(cv2.erode(wide_mask.astype(np.uint8), erode_kernel, iterations=1), xmap, ymap)
    home_frame = get_home_frame(l_red, u_red, unwrap_col)

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

        print(xmin, xmax)
        
    cwidth = (xmax - xmin)
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
    
def keypoint_height_calc(IK, int a0, int a2, int a3, int a4, int xS):    
    cdef double uc_w, vc_w, uc, vc
    cdef double rho, frho, lambda_est, z_est
    
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
    
# In theory should be able to parallelize matrix calcs for each feature, but for various reasons
# is not trivial. Leaving single calc functions here for legacy reasons, in case want to try
# again later
    
def Amat_single(siftdat,kp_retrieve, height_retrieve, kp_comp_sift):
    cdef double old_uc_w, old_vc_w
    cdef double old_uc, old_vc, h_pt
    cdef np.ndarray Apoint
    
    # calculate omnicam pixel positions using kp_retrieve
    old_uc_w = (kp_retrieve[siftdat.queryIdx].pt[0])*np.pi/(4*180)
    old_vc_w = (kp_retrieve[siftdat.queryIdx].pt[1])*300/360
    old_uc =  old_vc_w*np.sin(old_uc_w) 
    old_vc =  old_vc_w*np.cos(old_uc_w) 
    h_pt = height_retrieve[siftdat.queryIdx]
                        
    # create point-based matrices and append to existing structure 
    Apoint = np.array([[old_uc, -old_vc, -1/h_pt, 0],[old_vc, old_uc, 0, -1/h_pt]])
    return Apoint

def Bmat_single(siftdat, kp_retrieve, height_retrieve, kp_comp_sift):    
    cdef double new_uc_w, new_vc_w
    cdef double new_uc, new_vc
    cdef np.ndarray Bpoint

    new_uc_w = (kp_comp_sift[siftdat.trainIdx].pt[0])*np.pi/(4*180)
    new_vc_w = (kp_comp_sift[siftdat.trainIdx].pt[1])*300/360
    new_uc =  new_vc_w*np.sin(new_uc_w) 
    new_vc =  new_vc_w*np.cos(new_uc_w) 

    Bpoint = np.array([new_uc, new_vc])
    return Bpoint
    
def est_egomotion(sift_matches, InitialKeypoints, kp_comp_sift):    
    cdef np.ndarray Apoint, Bpoint
    cdef list Alist, Blist
    cdef double old_uc_w, old_vc_w
    cdef double old_uc, old_vc, h_pt
    cdef double new_uc_w, new_vc_w
    cdef double new_uc, new_vc
    cdef np.ndarray s_vec
    cdef double rotation, transx, transy
    cdef np.ndarray Qmat, U, s, V, Rmat
    
    Alist = []
    Blist = []
    kp_retrieve = InitialKeypoints.keypoints
    height_retrieve = InitialKeypoints.heights
    
    for siftdat in sift_matches[:15]:           
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
    
    

def dep_prob_calculator(int numblob, double whratio, np.ndarray minmax):
    cdef double blob_prob
    cdef double width_prob
    cdef np.ndarray retvec
    
    if numblob > 2: blob_prob = 0.2
    elif numblob > 1: blob_prob = 0.3
    else: blob_prob = 0.4

    if whratio > 1.5 or (min(minmax)<250) or (max(minmax)>500) : width_prob = 0.3
    elif whratio > 1.3: width_prob = 0.2
    else: width_prob = 0.1

    retvec= np.array([blob_prob, width_prob])

    return retvec

	
def localisation_calculator(np.ndarray locvec, np.ndarray probmatrix):
    cdef np.ndarray probvec
    cdef double probscalar
    
    probvec = np.dot(probmatrix, locvec)
    probscalar = np.sum(probvec)

    return probscalar



def omni_balance(list cp, np.ndarray omni_frame, np.ndarray mask, np.ndarray l_green, np.ndarray u_green):
    cdef np.ndarray bal_frame
    cdef np.ndarray dirvec_w
    
    cdef int sum_w_x = 0
    cdef int sum_w_y = 0    
    cdef double blob_area = 0.
    cdef double heading_angle = 0.

    # apply mask    
    # We use a downsampled mask conversion - bal_frame is now 300x300, adjust variables accordingly
    bal_frame = convertMask(omni_frame, l_green, u_green, mask, 1, erode_kernel, dilate_kernel)
    _, cnts, _ = cv2.findContours(bal_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if len(cnts)>0:
        # find contours with area > (some threshold) (there is a more elegant way of doing this)
        cont_max = max(cnts, key = cv2.contourArea)
        # y is vertical, x is horizontal
        cnts_lg = [c for c in cnts if cv2.contourArea(c)>800]        
        for cnt in cnts_lg:
            M = cv2.moments(cnt)
            cent_ang = np.arctan2(int(M['m10']/M['m00']) - cp[0]/2,int(M['m01']/M['m00']) - cp[1]/2)
            blob_area = cv2.contourArea(cnt)
            dirvec_w = np.array((np.sin(-cent_ang), np.cos(-cent_ang)),dtype = np.float)*blob_area/cv2.contourArea(cont_max)
                
            sum_w_y += dirvec_w[0]
            sum_w_x += dirvec_w[1]
            

        heading_angle = np.arctan2(sum_w_y, sum_w_x)
        if (0 < heading_angle < np.pi/2):
            heading_angle = heading_angle+np.pi
        elif (-np.pi/2 < heading_angle < 0):
            heading_angle = heading_angle - np.pi

        return heading_angle

    else:
        return 0



def omni_deposit(list cp, np.ndarray omni_frame, np.ndarray mask, np.ndarray l_green, np.ndarray u_green, np.ndarray xmap, np.ndarray ymap):
    cdef np.ndarray dep_frame_unwarp
    cdef np.ndarray box
    cdef double boxratio
    cdef int maxbox
    cdef int minbox
        
    # turn until we have one ROI of a sufficient size, with CoM within a central window 

    # CHECK SIZES: convertMask gives a 300x300 image, which may not map to the unwarp function. If this is a problem, use convertMask_full
    dep_frame_unwarp = unwarp(cv2.dilate(convertMask(omni_frame, l_green, u_green, mask, 1, erode_kernel, dilate_kernel), dilate_kernel_big, 1), xmap, ymap)
    _, cnts, _ = cv2.findContours(dep_frame_unwarp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # check how many segments larger than (threshold)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>500]

    if len(cnts_lg) > 0: 
        rect = cv2.minAreaRect(cnts_lg[0])
        box= np.int0(cv2.boxPoints(rect))
        maxbox = 2*max(box[:,0])
        minbox = 2*min(box[:,0])
        if (rect[2] > -45) and (rect[2] < 45):
            boxratio = rect[1][0]
            boxratio /= rect[1][1]
        else:
            boxratio = rect[1][1]
            boxratio /= rect[1][0]
        #else: # Ideally we would turn until wall and blob were merged, but this is hard to generalise
           
        M = cv2.moments(cnts_lg[0])                
        cy = int(M['m10']/M['m00'])
        heading_angle = (float(cy))*np.pi/180

        return len(cnts_lg), heading_angle, np.array([minbox, maxbox]), boxratio

    else:
        return 0, 0, np.array([0,0]), 0
        

def omni_home(list cp, np.ndarray omni_frame, np.ndarray mask, np.ndarray l_red, np.ndarray u_red):
    cdef np.ndarray back_frame    
    back_frame = convertMask(omni_frame, l_red, u_red, mask, 1, erode_kernel, dilate_kernel)
    _, cnts, _ = cv2.findContours(back_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnts_lg = [c for c in cnts if cv2.contourArea(c)> 50]
        if len(cnts_lg) > 0:
            return len(cnts_lg)

        else: return 0

    else: return 0


def leaving_home(list cp, np.ndarray omni_frame, np.ndarray wide_mask, np.ndarray l_red, np.ndarray u_red):
    cdef np.ndarray leave_frame    
    leave_frame = convertMask(omni_frame, l_red, u_red, wide_mask, 1, erode_kernel, dilate_kernel)
 
    _, cnts, _ = cv2.findContours(leave_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)> 900]

    return len(cnts_lg)


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


