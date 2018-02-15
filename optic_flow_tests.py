from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import numpy as np
import os

from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
from LeakyBot import LeakyBot
from leaky_nav_functions import *
from navImage import NavImage
from KeypointExpanded import KeypointExpanded
from PIL import Image
import imutils
from imutils.video import VideoStream

import time
import atexit

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

def shutdownLeaky():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)
    picam.stop()
    cv2.destroyAllWindows()

atexit.register(shutdownLeaky)


# Motor initialisation:
print("Initialising motor hat ...")
mh = Adafruit_MotorHAT(addr=0x60)
myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)
print("...done")

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

# ----------------
# set up mapping between spherical and unwarped images
print("Setting up unwarp map ...")
xmap, ymap = buildMap(600,600, 720, 360, 300, cp[0], cp[1])
print("...done")
# ----------------
# Set up camera
picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()
time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.1)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain
time.sleep(0.2)
# ----------------
# Establishing base image to home to: normally we would take a snapshot
print("Setting up features to track ...")
baseim = cv2.imread('./SIFT_testing/pattern_1/op_flow_10.png')
baseim = cv2.cvtColor(baseim, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(baseim, cv2.COLOR_BGR2GRAY)

wide_mask_init = wide_mask
wide_mask_init[gray>210] = 0
init_mask_erode = cv2.erode(wide_mask_init.astype(np.uint8), erode_kernel, iterations=1)

unwrap_base = unwarp(baseim, xmap, ymap)
unwrap_gray = unwarp(gray, xmap, ymap)
mask_unwrap = unwarp(init_mask_erode, xmap, ymap)

# MASK out ROI
lr_bound = 5
ur_bound = 25

l_red, u_red = boundary_estimate(unwrap_base, lr_bound, ur_bound, 100, 255, 100, 255, 15)

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

owidth = xmax - xmin
print("Bounds for feature extraction: ", xmin, xmax)
stripe_mask_corners = np.array([[xmin, 0], [xmin, 360], [xmax,360], [xmax,0]])
mask_shape = mask_unwrap.shape
stripe_mask = np.zeros((mask_shape))
cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
tracking_mask[tracking_mask>0] = 255

# Identify points: SIFT works better than surf, ultimately
sift = cv2.xfeatures2d.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(unwrap_gray, tracking_mask.astype(np.uint8))

# take all x coordinates of kp_surf
InitialKeypoints = KeypointExpanded(kp_sift)
# need to get mirror calibration data: a0, a2, a3
a0 = -188.44
a2 = 0.0072
a3 = -0.0000374
a4 = 0.0000000887
xS = 105

print("Establishing feature locations ...")

for kp_it in InitialKeypoints.keypoints:
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
    InitialKeypoints.add_height([z_est])


print("Starting Leaky ...")
leaky1 = LeakyBot(myMotor1, myMotor2)
leaky1.speed = 180
leaky1.direction = 'fwd'

print("Finished setup, preparing to home")
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Change to homing loop: speed up by keeping wide mask constant
running = True

# How to navigate:
# if we can't see any features (ie des_cop_surf = None), back up and turn a little

leaky1.cam_flag = 1 # redundant, but wev
fcount = 1
direction_weight = 0
ratio_weight = 0

while running:
    read_im = picam.read()
    compare_im = read_im[367:967, 536:1136,:]
    #img = Image.fromarray(compare_im)
    imname = './SIFT_testing/check_homing'
    imname += str(fcount)
    imname += '.jpg'
    #img.save(imname)
    fcount += 1
    comp_gray = cv2.cvtColor(compare_im, cv2.COLOR_BGR2GRAY)
    comp_gray_unwrap = unwarp(comp_gray, xmap, ymap)
    comp_unwrap = unwarp(compare_im, xmap, ymap)
    
    redframe = NavImage(comp_unwrap.copy())
    redframe.convertHsv()
    redframe.hsvMask(l_red, u_red)
    #img = Image.fromarray(redframe.frame)
    #img.save(imname)

    redframe.erodeMask(erode_kernel, 1)
    redframe.dilateMask(dilate_kernel, 1)
    _, cnts, _ = cv2.findContours(redframe.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_lg = [c for c in cnts if cv2.contourArea(c)>300]
    xmin = 600
    xmax = 0
    ymin = 0
    for c in cnts_lg:
        x, y, w, h = cv2.boundingRect(c)
        if y > ymin: 
            ymin = y
            hstore = h
        if (x > 50) and (x< 650):
            if x+w > xmax: xmax = x+w
            if x < xmin : xmin = x

    cwidth = xmax - xmin
    delta = float(cwidth)/float(owidth)
    if delta < 1:
        ratio_weight = 0
        smask_y = int(delta*180)
    elif ratio_weight > 0.2: 
        smask_y = 180
        ratio_weight = ratio_weight
    else:
        smask_y = 180
        ratio_weight = 0.2

    if delta > 1: ratio_weight += 0.1

    if hstore > 60: height_weight = 0.3
    else: height_weight = 0

    print(delta, hstore)

    stripe_mask_corners = np.array([[xmin, smask_y], [xmin, ymin], [xmax,ymin], [xmax,smask_y]])
    mask_shape = mask_unwrap.shape
    stripe_mask = np.zeros((mask_shape))
    cv2.fillConvexPoly(stripe_mask, stripe_mask_corners, 1)

    tracking_mask = cv2.bitwise_and(mask_unwrap.astype(np.uint8), stripe_mask.astype(np.uint8))
    tracking_mask[tracking_mask>0] = 255
    
    img = Image.fromarray(tracking_mask)
    img.save(imname)
    checktime = time.time()
    kp_comp_sift, des_comp_sift = sift.detectAndCompute(comp_gray_unwrap, tracking_mask.astype(np.uint8))
    #print(time.time() - checktime)

    if (not (des_comp_sift is None)) :
        sift_matches = bf.match(des_sift, des_comp_sift) 
        sift_matches = sorted(sift_matches, key= lambda x:x.distance)
        match_image = cv2.drawMatches(unwrap_base, kp_sift, comp_unwrap, kp_comp_sift, sift_matches[:15], None, flags=2)
        Alist = []
        Blist = []
        for siftdat in sift_matches[:15]:
            # retrieve data:
            kp_retrieve = InitialKeypoints.keypoints
            height_retrieve = InitialKeypoints.heights
            # Put this stuff in a subfunction
            
            # as before: calculate omnicam pixel positions using kp_retrieve
            old_uc_w = (kp_retrieve[siftdat.queryIdx].pt[0])*np.pi/(4*180)
            old_vc_w = (kp_retrieve[siftdat.queryIdx].pt[1])*300/360
            old_uc =  old_vc_w*np.sin(old_uc_w) 
            old_vc =  old_vc_w*np.cos(old_uc_w) 
            h_pt = height_retrieve[siftdat.queryIdx]
            
            new_uc_w = (kp_comp_sift[siftdat.trainIdx].pt[0])*np.pi/(4*180)
            new_vc_w = (kp_comp_sift[siftdat.trainIdx].pt[1])*300/360
            new_uc =  new_vc_w*np.sin(new_uc_w) 
            new_vc =  new_vc_w*np.cos(new_uc_w) 
            
            # Question: are these actual positions (including cp) or normalized positions (cp = 0) ?
            # suspect normalised.
            
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
        # Some wrap issues, I think:
        rotation = np.arctan(Rmat[1,0]/Rmat[0,0])
        transx = s_vec[2]
        transy = s_vec[3]
        print("Estimated egomotion:" , rotation*180/np.pi, transx, transy)
        # Head left: left goes backwards, right goes forward
        # homography calculation works ok for small shifts, less well for large ones
        if abs(rotation*180/np.pi) > 5:
            direction_weight = 0
            if rotation > 0:
                print('turning left')
                leaky1.cam_flag = 1
                leaky1.direction = 'left'
                leaky1.set_motor_values(leaky1.speed, leaky1.speed, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.FORWARD)
                time.sleep(0.08)
                leaky1.auto_set_motor_values(0,0)
                
            elif rotation < 0:
                print('turning right')
                leaky1.cam_flag = 0
                leaky1.direction = 'right'
                leaky1.set_motor_values(leaky1.speed, leaky1.speed, Adafruit_MotorHAT.FORWARD, Adafruit_MotorHAT.BACKWARD)
                time.sleep(0.08)
                leaky1.auto_set_motor_values(0,0)
                
        elif transx < -5:
            print('going forward')
            leaky1.direction = 'fwd'
            leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
            time.sleep(0.2)
            leaky1.auto_set_motor_values(0,0)
              
        if (abs(rotation*180/np.pi) < 10) and ((ratio_weight > 0) or (height_weight>0)): direction_weight += 0.1
        else: direction_weight = 0
        
        print("Weightings: ", ratio_weight, height_weight, direction_weight)
        weight_array = np.array([ratio_weight, height_weight, direction_weight])

        if (np.sum(weight_array)> 0.5 and (ratio_weight >0 and height_weight>0)):
            print("Think I'm home")
            running = False
        
    else:
        print("cannot find relevant features, backing up")
        leaky1.cam_flag = 1
        leaky1.direction = 'revturn'
        leaky1.auto_set_motor_values(leaky1.speed, leaky1.speed)
        time.sleep(0.1)
        leaky1.auto_set_motor_values(0,0)
        

        
   
