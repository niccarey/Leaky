import cv2
import numpy as np
import os

from leaky_nav_functions import *
from navImage import NavImage

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
    

# Define omnicam masks:
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

# May need to unwarp image to use optic flow? This might also improve feature matching
# get initial image
baseim = cv2.imread('op_flow_10.png')
gray = cv2.cvtColor(baseim, cv2.COLOR_BGR2GRAY)

wide_mask_init = wide_mask
wide_mask_init[gray>220] = 0
init_mask_erode = cv2.erode(wide_mask_init.astype(np.uint8), erode_kernel, iterations=1)
#init_mask_erode[init_mask_erode>0] = 255

xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])

# Unwarp image and mask
unwrap_base = unwarp(baseim, xmap, ymap)
unwrap_gray = unwarp(gray, xmap, ymap)
mask_unwrap = unwarp(init_mask_erode, xmap, ymap)

# Method: convert to HSV, use usual method to obtain RED and GREEN blobs
# May want to check histogram output of direct frames, this looks shifted compared to usual

lr_bound = 90
ur_bound = 120

l_red, u_red = boundary_estimate(baseim, lr_bound, ur_bound, 80, 255, 40, 230, 15)

redframe = NavImage(unwrap_base.copy())
redframe.convertHsv()
redframe.hsvMask(l_red, u_red)

# TO DO: code a red/green filter subfunction that quickly saves snapshots
# use this as a sanity check before running leaky_run as suspect poor tuning may
# have caused many of the previous problems

#cv2.imshow("testing mask", greenframe.frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# expand drastically and combine masks
redframe.erodeMask(erode_kernel, 1)
redframe.dilateMask(dilate_kernel, 1)

# TODO:  use redframe bounding box to derive search mask for patterns
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
# create corners
stripe_mask_corners = np.array([[xmin, 0], [xmin, 360], [xmax,360], [xmax,0]])

# Generate 2D unwarped mask that includes only areas between red stripes
mask_shape = mask_unwrap.shape
stripe_mask = np.zeros((mask_shape))
cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
tracking_mask[tracking_mask>0] = 255
#cv2.imshow("test red mask", tracking_mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sift = cv2.xfeatures2d.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

surf = cv2.xfeatures2d.SURF_create()
kp_surf, des_surf = surf.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

imgdisp  = cv2.drawKeypoints(unwrap_base, kp_surf, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Features', imgdisp)

cv2.waitKey(0)

imgdisp  = cv2.drawKeypoints(unwrap_base, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Features', imgdisp)
cv2.waitKey(0)

# Let's try Good Features To Track
#feature_params = dict( maxCorners = 100, qualityLevel = 0.9, minDistance = 7, blockSize = 7)
#p_feat = cv2.goodFeaturesToTrack(unwrap_gray, mask=tracking_mask.astype(np.uint8), **feature_params)
color = np.random.randint(0,255,(100,3))

#i=0
#for p in p_feat:
#    c = p[0,0]
#    d = p[0,1]
#    framedisp = unwrap_base.copy()
#    framedisp = cv2.circle(framedisp, (c,d), 5, color[i].tolist(), -1)
#    i += 1
    
#cv2.imshow("Features", framedisp)
#cv2.waitKey(0)
cv2.destroyAllWindows()

# Brute force matching:
# knn not great. FLANN may be faster, but same issues
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

FLANN_INDEX_KDTREE=1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# FLANN knn is comparable to brute force non-knn, but when results are wrong they have more of a bias.

for file in files("."):
    print(file)
    compare_im = cv2.imread(file)
    #comp_recol = cv2.cvtColor(compare_im, cv2.COLOR_RGB2BGR)
    
    # Set up masks in polar coordinates (easier)
    # Do I need to exclude white now?
    comp_gray = cv2.cvtColor(compare_im, cv2.COLOR_BGR2GRAY)
    
    wide_mask_comp = wide_mask
    wide_mask_comp[comp_gray>210] = 0    
    mask_erode = cv2.erode(wide_mask_comp.astype(np.uint8), erode_kernel, iterations=1)
    #mask_erode[mask_erode>0] = 255
    
    comp_mask_unwrap = unwarp(mask_erode, xmap, ymap)
    comp_gray_unwrap = unwarp(comp_gray, xmap, ymap)
    comp_unwrap = unwarp(compare_im, xmap, ymap)
    
    # Score with restrictions:
    redframe = NavImage(comp_unwrap.copy())
    redframe.convertHsv()
    redframe.hsvMask(l_red, u_red)
    
    # expand drastically and combine masks
    redframe.erodeMask(erode_kernel, 1)
    redframe.dilateMask(dilate_kernel, 1)
    
    _, cnts, _ = cv2.findContours(redframe.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>300]
    xmin = 600
    xmax = 0
    for c in cnts_lg:
        x, y, w, h = cv2.boundingRect(c)
        if (x > 100) and (x<550):
            if x+w > xmax: xmax = x+w
            if x < xmin: xmin = x
    
    # create corners
    stripe_mask_corners = np.array([[xmin, 0], [xmin, 360], [xmax,360], [xmax,0]])
    mask_shape = mask_unwrap.shape
    stripe_mask = np.zeros((mask_shape))
    cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

    tracking_mask = cv2.bitwise_and(comp_mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
    tracking_mask[tracking_mask>0] = 255
    
    kp_comp_surf, des_comp_surf = surf.detectAndCompute(comp_gray_unwrap, tracking_mask.astype(np.uint8))
    kp_comp_sift, des_comp_sift = sift.detectAndCompute(comp_gray_unwrap, tracking_mask.astype(np.uint8))
        
    if (not (des_comp_surf is None)) :
        surf_matches = bf.match(des_surf, des_comp_surf) #.astype(np.uint32), des_comp.astype(np.uint32))
        #surf_matches = flann.knnMatch(des_surf, des_comp_surf, k=2)
        #surf_matchesMask = [[0,0] for i in xrange(len(surf_matches))]
        #for i, (m,n) in enumerate(surf_matches):
        #    if m.distance < 0.7*n.distance:
        #        surf_matchesMask[i] = [1,0]
        surf_matches = sorted(surf_matches, key= lambda x:x.distance)
            
        #draw_surf_params = dict(matchColor = color[np.random.randint(0,100)], singlePointColor = (255,0,255), matchesMask = surf_matchesMask, flags = 0)
    
    # BF knn matching
    #blue_matches = bf.knnMatch(des_blue.astype(np.uint8), des_comp_b.astype(np.uint8), k=2)
    #green_matches = bf.knnMatch(des_green.astype(np.uint8), des_comp_g.astype(np.uint8), k=2)

    #matches = blue_matches
    #matches.extend(green_matches)
    
    # ratio_test:
    #good = []
    #for m,n in matches:
    #    if m.distance < 0.75*n.distance :
    #        good.append([m])
            
                
        match_image = cv2.drawMatches(unwrap_base, kp_surf, comp_unwrap, kp_comp_surf, surf_matches[:20], None, flags=2)
        #match_image = cv2.drawMatchesKnn(baseim, kp, compare_im, kp_comp, good, None, flags=2)
        #match_image = cv2.drawMatchesKnn(unwrap_base, kp_surf, comp_unwrap, kp_comp_surf, surf_matches, None, **draw_surf_params)

        cv2.imshow('check match', match_image)
        cv2.waitKey(0)

    if (not (des_comp_sift is None)) :
    #    sift_matches = bf.match(des_sift, des_comp_sift) #.astype(np.uint32), des_comp.astype(np.uint32))
    #    sift_matches = sorted(sift_matches, key= lambda x:x.distance)
        # FLANN matching
        sift_matches = flann.knnMatch(des_sift, des_comp_sift, k=2)
        sift_matchesMask = [[0,0] for i in xrange(len(sift_matches))]
        # Similar ratio test
        for i, (m,n) in enumerate(sift_matches):
            if m.distance < 0.7*n.distance:
                sift_matchesMask[i] = [1,0]
            
        draw_sift_params = dict(matchColor = (0,0,255), singlePointColor = (255,0,255), matchesMask = sift_matchesMask, flags = 0)
                
        #match_image = cv2.drawMatches(unwrap_base, kp_sift, comp_unwrap, kp_comp_sift, sift_matches[:20], None, flags=2)
        #match_image = cv2.drawMatchesKnn(baseim, kp, compare_im, kp_comp, good, None, flags=2)
        match_image = cv2.drawMatchesKnn(unwrap_base, kp_sift, comp_unwrap, kp_comp_sift, sift_matches, None, **draw_sift_params)

        cv2.imshow('check match', match_image)
        #cv2.imshow('check green match', match_image_g)
        cv2.waitKey(0)

