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
r_inner = 180;
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
baseim = cv2.imread('SIFTtuning_010.png')
gray = cv2.cvtColor(baseim, cv2.COLOR_BGR2GRAY)

wide_mask_init = wide_mask
wide_mask_init[gray>160] = 0
init_mask_erode = cv2.erode(wide_mask_init.astype(np.uint8), erode_kernel, iterations=1)
init_mask_erode[init_mask_erode>0] = 255

# Unwrap mask: This is super slow, need to speed up A LOT
# Look into array transformations
# We can pre-code a mapping between pixel source and pixel destination, to speed things up
xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])

# Now try unwarping mask:
mask_unwrap = unwarp(init_mask_erode, xmap, ymap)

# Color information is lost in sift and surf (possibly others?) - our main features involve color. So try:
# What happens when we unwrap a color image?
unwrap_base = unwarp(baseim, xmap, ymap)


# Ceiling and wall features are not turning out to be useful. What happens 
# if we use an (inflated) HSV-based mask on the colour images to keep colour matching relevant?

# Method: convert to HSV, use usual method to obtain RED and GREEN blobs
# May want to check histogram output of direct frames, this looks shifted compared to usual
lg_bound = 40
ug_bound = 80

lr_bound = 90
ur_bound = 120

l_red, u_red = boundary_estimate(baseim, lr_bound, ur_bound, 80, 255, 40, 230, 15)
l_green, u_green = boundary_estimate(baseim, lg_bound, ug_bound, 60, 255, 0, 255, 25)

redframe = NavImage(unwrap_base.copy())
redframe.convertHsv()
redframe.hsvMask(l_red, u_red)

greenframe = NavImage(unwrap_base.copy())
greenframe.convertHsv()
greenframe.hsvMask(l_green, u_green)

# TO DO: code a red/green filter subfunction that quickly saves snapshots
# use this as a sanity check before running leaky_run as suspect poor tuning may
# have caused many of the previous problems

#cv2.imshow("testing mask", greenframe.frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# expand drastically and combine masks
redframe.erodeMask(erode_kernel, 1)
greenframe.erodeMask(erode_kernel, 1)
redframe.dilateMask(dilate_kernel, 1)
greenframe.dilateMask(dilate_kernel, 1)
# TODO: rewrite some of the nav code so it's more sensible, using bitwise masking
color_mask = cv2.bitwise_or(redframe.frame, greenframe.frame)

# add mask to unwrapped mask, use for detection
mask_unwrap = cv2.bitwise_and(color_mask, mask_unwrap)
cv2.imshow('Check mask', mask_unwrap)
cv2.waitKey(0)
cv2.destroyAllWindows()

blue, green, red = cv2.split(unwrap_base)

# Stick with SIFT for now:
#sift = cv2.xfeatures2d.SIFT_create()
#kp_blue, des_blue= sift.detectAndCompute(blue, mask_unwrap.astype(np.uint8))
#kp_green, des_green = sift.detectAndCompute(green, mask_unwrap.astype(np.uint8))


surf = cv2.xfeatures2d.SURF_create()
kp_blue, des_blue = surf.detectAndCompute(blue, mask_unwrap.astype(np.uint8))
kp_green, des_green = surf.detectAndCompute(green, mask_unwrap.astype(np.uint8))

kp = kp_blue
kp.extend(kp_green)

# Fast detection isn't very good 

# ORB might be faster?
#orb = cv2.ORB_create()
#kp = orb.detect(gray, wide_mask.astype(np.uint8))

imgdisp  = cv2.drawKeypoints(unwrap_base, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT features', imgdisp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Brute force matching:
# knn not great
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#FLANN_INDEX_KDTREE=1
#index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#search_params = dict(checks=50)
#flann = cv2.FlannBasedMatcher(index_params, search_params)


for file in files("."):
    print(file)
    compare_im = cv2.imread(file)
    unwrap_comp = unwarp(compare_im, xmap, ymap)
    #comp_recol = cv2.cvtColor(compare_im, cv2.COLOR_RGB2BGR)
    
    # Set up masks in polar coordinates (easier)
    comp_gray = cv2.cvtColor(compare_im, cv2.COLOR_BGR2GRAY)
    
    wide_mask_comp = wide_mask
    wide_mask_comp[comp_gray>160] = 0    
    mask_erode = cv2.erode(wide_mask_comp.astype(np.uint8), erode_kernel, iterations=1)
    mask_erode[mask_erode>0] = 255
    comp_mask_unwrap = unwarp(mask_erode, xmap, ymap)
    
    # Now: use color-based masking as above
    redframe = NavImage(unwrap_comp.copy())
    redframe.convertHsv()
    redframe.hsvMask(l_red, u_red)

    greenframe = NavImage(unwrap_comp.copy())
    greenframe.convertHsv()
    greenframe.hsvMask(l_green, u_green)
    
    # expand drastically and combine masks
    redframe.erodeMask(erode_kernel, 1)
    greenframe.erodeMask(erode_kernel, 1)
    redframe.dilateMask(dilate_kernel, 1)
    greenframe.dilateMask(dilate_kernel, 1)
    color_mask = cv2.bitwise_or(redframe.frame, greenframe.frame)

    # add mask to unwrapped mask, use for detection
    comp_mask_unwrap = cv2.bitwise_and(color_mask, comp_mask_unwrap)

    blue,green,_ = cv2.split(unwrap_comp)
    #kp_comp_b, des_comp_b = sift.detectAndCompute(blue, comp_mask_unwrap.astype(np.uint8))
    #kp_comp_g, des_comp_g = sift.detectAndCompute(green, comp_mask_unwrap.astype(np.uint8))
    
    kp_comp_b, des_comp_b = surf.detectAndCompute(blue, comp_mask_unwrap.astype(np.uint8))
    kp_comp_g, des_comp_g = surf.detectAndCompute(green, comp_mask_unwrap.astype(np.uint8))
    
    
    kp_comp = kp_comp_b
    kp_comp.extend(kp_comp_g)
    
    # FLANN matching
    #blue_matches = flann.knnMatch(des_blue, des_comp_b, k=2)
    #green_matches = flann.knnMatch(des_green, des_comp_g, k=2)
    #
    #matches = blue_matches
    #matches.extend(green_matches)
    #
    #matchesMask = [[0,0] for i in xrange(len(matches))]
    
    # Similar ratio test
    #for i, (m,n) in enumerate(matches):
    #    if m.distance < 0.7*n.distance:
    #        matchesMask[i] = [1,0]
    #        
    #draw_params = dict(matchColor = (0,0,255), singlePointColor = (255,0,255), matchesMask = matchesMask, flags = 0)
    

    # bf matching
    if (not (des_comp_b is None)) and (not(des_comp_g) is None):
        blue_matches = bf.match(des_blue, des_comp_b) #.astype(np.uint32), des_comp.astype(np.uint32))
        green_matches = bf.match(des_green, des_comp_g )#.astype(np.uint32), des_comp.astype(np.uint32))

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
            
        blue_matches = sorted(blue_matches, key= lambda x:x.distance)
        green_matches = sorted(green_matches, key= lambda x:x.distance)

        matches = blue_matches
        matches.extend(green_matches)
    
        match_image = cv2.drawMatches(unwrap_base, kp, unwrap_comp, kp_comp, matches[:10], None, flags=2)
        #match_image = cv2.drawMatchesKnn(baseim, kp, compare_im, kp_comp, good, None, flags=2)
        #match_image = cv2.drawMatchesKnn(baseim, kp, compare_im, kp_comp, matches, None, **draw_params)

        cv2.imshow('check match', match_image)
        #cv2.imshow('check green match', match_image_g)
        cv2.waitKey(0)
    
