import cv2
import numpy as np
import os

from leaky_nav_functions import *
from navImage import NavImage
from KeypointExpanded import KeypointExpanded

# get list of files

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)):
            yield file


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

# use either sift or surf point selection with BF matching (surf may be slightly more consistent)
    
# We need to unwrap images, unfortunately(!)]
# Remap points to polar coordinates to calculate rotation vs translation? Maybe not.

# DEFINE MASKS for input images
cp = [ 300, 300 ]
r_out = 298;
r_inner = 150;
r_norim = 295;

poly_front = np.array([cp, [20, 1], [600,1]])
poly_back = np.array([cp, [1, 600], [600,600], [600,430]])
poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
poly_right = np.array([cp, [600, 1], [600 ,1], [600, 430]])

sides_mask, front_mask, wide_mask = define_masks([600,600], cp, r_out, r_inner, r_norim, poly_front, poly_back, poly_left, poly_right)
erode_kernel = np.ones((7,7), np.uint8)
dilate_kernel = np.ones((19,19), np.uint8)

# the Lucas-kanade advantage is that it looks at pixels surrounding an identified point.
# But hunting for these points MAY assume only small jumps in motion. Because we are homing via snapshot
# we may have large jumps
xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])

# Get initial image
baseim = cv2.imread('op_flow_10.png')
gray = cv2.cvtColor(baseim, cv2.COLOR_BGR2GRAY)

wide_mask_init = wide_mask
wide_mask_init[gray>220] = 0
init_mask_erode = cv2.erode(wide_mask_init.astype(np.uint8), erode_kernel, iterations=1)

unwrap_base = unwarp(baseim, xmap, ymap)
unwrap_gray = unwarp(gray, xmap, ymap)
mask_unwrap = unwarp(init_mask_erode, xmap, ymap)

# MASK out ROI
lr_bound = 90
ur_bound = 120

l_red, u_red = boundary_estimate(unwrap_base, lr_bound, ur_bound, 100, 255, 80, 255, 15)

redframe = NavImage(unwrap_base.copy())
redframe.convertHsv()
redframe.hsvMask(l_red, u_red)
redframe.erodeMask(erode_kernel, 1)
redframe.dilateMask(dilate_kernel, 1)

_, cnts, _ = cv2.findContours(redframe.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts_lg = [c for c in cnts if cv2.contourArea(c)>300]
xmin = 720
xmax = 0
for c in cnts_lg:
    x, y, w, h = cv2.boundingRect(c)
    if x > 100 and x < 550:
        if x+w > xmax: xmax = x+w
        if x < xmin: xmin = x


print(xmin, xmax)
owidth = xmax - xmin
stripe_mask_corners = np.array([[xmin, 0], [xmin, 360], [xmax,360], [xmax,0]])
mask_shape = mask_unwrap.shape
stripe_mask = np.zeros((mask_shape))
cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
tracking_mask[tracking_mask>0] = 255

# Identify points
#surf = cv2.xfeatures2d.SURF_create()
#kp_surf, des_surf = surf.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))


sift = cv2.xfeatures2d.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

# take all x coordinates of kp_surf
InitialKeypoints = KeypointExpanded(kp_sift)
# need to get mirror calibration data: a0, a2, a3
a0 = -188.44
a2 = 0.0072
a3 = -0.0000374
a4 = 0.0000000887
xS = 155


for kp_it in InitialKeypoints.keypoints:
    uc_w = (kp_it.pt[0])*np.pi/(4*180)
    vc_w = kp_it.pt[1]*300/360
    # Convert to omnicam pixel positions (I've fucked this up)
    uc =  vc_w*np.sin(uc_w)
    vc =  vc_w*np.cos(uc_w)
    #print(uc,vc)
    rho = np.sqrt(np.square(uc) + np.square(vc ))
    frho = a0 + a2*np.square(rho) + a3*np.power(rho,3) + a4*np.power(rho,4)
    lambda_est = xS/(uc)
    # invert because calibration assumes an upside-down mirror
    z_est = -(lambda_est*frho)
    InitialKeypoints.add_height([z_est])
    

imgdisp  = cv2.drawKeypoints(unwrap_base, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Features', imgdisp)
cv2.waitKey(0)


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

for file in files("."):
    print(file)
    compare_im = cv2.imread(file)
    comp_gray = cv2.cvtColor(compare_im, cv2.COLOR_BGR2GRAY)
    
    #wide_mask_comp = wide_mask
    #wide_mask_comp[comp_gray>210] = 0
    #mask_erode = cv2.erode(wide_mask_comp.astype(np.uint8), erode_kernel, iterations=1)
    #comp_mask_unwrap = unwarp(mask_erode, xmap, ymap)
    comp_gray_unwrap = unwarp(comp_gray, xmap, ymap)
    comp_unwrap = unwarp(compare_im, xmap, ymap)
    
    redframe = NavImage(comp_unwrap.copy())
    redframe.convertHsv()
    redframe.hsvMask(l_red, u_red)
    
    # expand drastically and combine masks
    redframe.erodeMask(erode_kernel, 1)
    redframe.dilateMask(dilate_kernel, 1)

    _, cnts, _ = cv2.findContours(redframe.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>500]
    xmin = 600
    xmax = 0
    ymin = 0
    for c in cnts_lg:
        x, y, w, h = cv2.boundingRect(c)
        if y > ymin:
            ymin = y
        if (x > 50) and (x<650):
            if x+w > xmax: xmax = x+w
            if x < xmin: xmin = x
    
    
    cwidth = xmax - xmin
    delta = float(cwidth)/float(owidth)
    
    if delta < 1:
        smask_y = int(delta*180)
    else:
        smask_y = 180
        
    print("extrema:", xmin, xmax, ymin)
        
    stripe_mask_corners = np.array([[xmin, smask_y], [xmin, ymin], [xmax,ymin], [xmax,smask_y]])
    mask_shape = mask_unwrap.shape
    stripe_mask = np.zeros((mask_shape))
    cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

    tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
    tracking_mask[tracking_mask>0] = 255
    cv2.imshow("check mask", tracking_mask)
    cv2.waitKey(0)

    
    #kp_comp_surf, des_comp_surf = surf.detectAndCompute(comp_gray_unwrap, tracking_mask.astype(np.uint8))
    kp_comp_sift, des_comp_sift = sift.detectAndCompute(comp_gray_unwrap, tracking_mask.astype(np.uint8))

    if (not (des_comp_sift is None)) :
        surf_matches = bf.match(des_sift, des_comp_sift) 
        surf_matches = sorted(surf_matches, key= lambda x:x.distance)
        match_image = cv2.drawMatches(unwrap_base, kp_sift, comp_unwrap, kp_comp_sift, surf_matches[:15], None, flags=2)
        Alist = []
        Blist = []
        for surfdat in surf_matches[:15]:
            # retrieve data:
            kp_retrieve = InitialKeypoints.keypoints
            height_retrieve = InitialKeypoints.heights
            # Put this stuff in a subfunction
            
            # as before: calculate omnicam pixel positions using kp_retrieve
            old_uc_w = (kp_retrieve[surfdat.queryIdx].pt[0])*np.pi/(4*180)
            old_vc_w = (kp_retrieve[surfdat.queryIdx].pt[1])*300/360
            old_uc =  old_vc_w*np.sin(old_uc_w) 
            old_vc =  old_vc_w*np.cos(old_uc_w) 
            h_pt = height_retrieve[surfdat.queryIdx]
            
            new_uc_w = (kp_comp_sift[surfdat.trainIdx].pt[0])*np.pi/(4*180)
            new_vc_w = (kp_comp_sift[surfdat.trainIdx].pt[1])*300/360
            new_uc =  new_vc_w*np.sin(new_uc_w) 
            new_vc =  new_vc_w*np.cos(new_uc_w) 
            
            # Question: are these actual positions (including cp) or normalized positions (cp = 0) ?
            # suspect normalised.
            
            # create point-based matrices and append to existing structure 
            Apoint = np.array([[old_uc, -old_vc, -1/h_pt, 0],[old_vc, old_uc, 0, -1/h_pt]])
            Bpoint = np.array([new_uc, new_vc])

            Alist.extend(Apoint,)
            Blist.extend(Bpoint,)
            
            
            #xdiff = kp_surf[surfdat.queryIdx].pt[0] - kp_comp_surf[surfdat.trainIdx].pt[0]
            #ydiff = kp_surf[surfdat.queryIdx].pt[1] - kp_comp_surf[surfdat.trainIdx].pt[1]
            #print(xdiff, ydiff)
            
        Alist = np.array(Alist)
        Blist = np.array(Blist)
        print(Alist.shape, Blist.shape)
        s_vec = np.dot(np.linalg.pinv(Alist), np.transpose(Blist))
        Qmat = np.array([[s_vec[0], -s_vec[1]], [s_vec[1], s_vec[0]]])
        U, s, V = np.linalg.svd(Qmat)
        Rmat = np.dot(U, np.transpose(V))
        # Some wrap issues, I think:
        rotation = np.arctan(Rmat[1,0]/Rmat[0,0])
        transx = s_vec[2]
        transy = s_vec[3]
        print("Estimated egomotion:" , rotation*180/np.pi, transx, transy)
        # How to handle? Fix heading angle, then x translation. Hopefully y takes care of itself.
        # heading angle looks pretty good, translation looks OK (haven't really tested it)
        cv2.imshow('check match', match_image)
        cv2.waitKey(0)

            
        
   
