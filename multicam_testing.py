from picamera import PiCamera
from picamera.array import PiRGBArray
from Adafruit_MotorHAT import Adafruit_MotorHAT
import numpy as np 
import threading
import skimage
import os
from skimage.measure import structural_similarity as ssim

from navImage import NavImage


import imutils
from imutils.video import VideoStream
import cv2
import time
import atexit


mh = Adafruit_MotorHAT(addr=0x60)

myMotor1 = mh.getMotor(1)
myMotor2 = mh.getMotor(3)

# initialise pycam
g = (1.5,2.1)


# Initialize webcam

webcam = VideoStream(src=0).start()
picam = VideoStream(usePiCamera=True, resolution=(640,480)).start()

time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.1)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain

os.system('v4l2-ctl --set-ctrl=white_balance_temperature_auto=0')
os.system('v4l2-ctl --set-ctrl=white_balance_temperature=2800')
os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
os.system('v4l2-ctl --set-ctrl=exposure_absolute=150')
os.system('v4l2-ctl --set-ctrl=brightness=0')    



# manual white balancing not possible with this driver (?)
# should be ok as long as lighting conditions relatively static
#webcam.stream.set(18,0.5)

# Set up templates and image processing constants
erode_kernel = np.ones((7,7), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)

l_red_left = np.array([170, 180, 0])
u_red_left = np.array([180, 255, 255])
l_red_right = np.array([0, 200, 0])
u_red_right = np.array([10, 255, 255])

l_blue_left = np.array([40, 100, 0])
u_blue_left = np.array([70, 255, 255])
l_blue_right = np.array([50, 100, 0])
u_blue_right = np.array([90, 255, 255])

# Matched filtering:
kernsize = 17
temp_size = 30
temp_blur_size = 11
corner_template = np.zeros((120,temp_size))
corner_template[:,1:15] = 1

left_template = np.array(255*cv2.GaussianBlur(corner_template, (kernsize, kernsize),0), dtype=np.uint8)
right_template = cv2.flip(left_template, 1)

#cv2.imshow("WTF", left_template)

cX_left, cY_left = [0, 0]
cX_right, cY_right = [0, 0]

navflow_left = np.zeros((480,640))
navflow_right = np.zeros((480,640))

def turnOffMotors():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)    

def create_masked_im(nav_frame, l_bound, u_bound):
    nav_image = NavImage(nav_frame) 
    nav_image.convertHsv()     
    nav_image.hsvMask(l_bound, u_bound)
    return(nav_image.frame)
    # additional operations: probably don't need
    #nav_image.erodeMask(erode_kernel, 1) 
    #nav_image.dilateMask(dilate_kernel, 1)
    #blur_im = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)

    #return blur_im
    
    
def find_contours(full_image):
    dist_transform = cv2.distanceTransform(full_image, cv2.DIST_L2,5)

    _, region_of_interest = cv2.threshold(dist_transform, 0.12*dist_transform.max(),255,0)
    roi = np.uint8(region_of_interest)
    #cv2.imshow("Slow catchup", roi)
    _, contours, heirarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cXList = []
    cYList = []
    for cntr in contours:
        if cv2.contourArea(cntr) > 1100:
            # get centroid of each contour
            mments = cv2.moments(cntr)
            cX = int(mments['m10']/mments['m00'])
            cY = int(mments['m01']/mments['m00'])    
            cXList.append(cX)
            cYList.append(cY)
            
    print("Found centroids: (x-loc) ", cXList)
    
    
def find_edge(nav_frame, l_bound, u_bound, cam_flag, location):
    global navflow_left, navflow_right
    global cX_left, cY_left
    global cX_right, cY_right
    
    nav_image = NavImage(nav_frame) 
    nav_image.convertHsv()     
    nav_image.hsvMask(l_bound, u_bound)
    
    
    # additional operations: probably don't need
    nav_image.erodeMask(erode_kernel, 1) 
    nav_image.dilateMask(dilate_kernel, 1)
    blur_im = np.array(cv2.GaussianBlur(nav_image.frame, (temp_blur_size, temp_blur_size), 0), dtype=np.uint8)

    mments = cv2.moments(blur_im)
    if (mments['m00'] > 0):
        cX = int(mments['m10']/mments['m00'])
        cY = int(mments['m01']/mments['m00'])
        if cam_flag:
            cX_left = cX#max_loc[0]+temp_size//2 #half template size
            cY_left = cY#max_loc[1]
            #cv2.imshow("Left filter", blur_im)
            print("cX_left, ", cX_left)
            
        else:
            cX_right = cX#max_loc[0]+temp_size//2 #half template size
            cY_right = cY#max_loc[1]
            #print("cX_right, " , cX_right)
    
    #if cam_flag: #(direction == 'left'):
    #    template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), left_template, cv2.TM_CCORR_NORMED)
    #    navflow_left = blur_im
        
    #else:
    #    template_match = cv2.matchTemplate(np.array(nav_image.frame, dtype=np.uint8), right_template, cv2.TM_CCORR_NORMED)
    #    navflow_right = blur_im
    
    # Location
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    # only update if we're moderately certain about the edge
    #if not cam_flag:
    #    print(max_val)
        
    #if max_val > 0.66:
    #    found_block_flag = 1
    


    #else:
    #    found_block_flag = 0
        


running = True
fcount = 1


des_speed = 140
myMotor1.setSpeed(des_speed)
myMotor2.setSpeed(des_speed)

# note: include motor trim values to compensate for any bias (have to test without tethering)

wheel_count = 1


left_balance_check = 0
right_balance_check = 0

# Set navigation thresholds:
found_edge_threshold = 350
back_up_threshold = 520


#for frame in picam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
while running:


    # two problems: slow image processing (can thread)
    # second: how do we tell when to stop?

    if fcount > 20:
        # give video frames time to start properly
        if (left_balance_check < 25) and (right_balance_check < 25):
            if wheel_count == 10:
                myMotor1.run(Adafruit_MotorHAT.FORWARD)
                myMotor2.run(Adafruit_MotorHAT.FORWARD)
                wheel_count += 1

            elif wheel_count == 15:
                myMotor1.run(Adafruit_MotorHAT.RELEASE)
                myMotor2.run(Adafruit_MotorHAT.RELEASE)
                wheel_count = 1

      
            else:
                wheel_count += 1

        else:                                                                                                                                                  
            # stop, enter waiting state
            print("reached home")
            myMotor1.run(Adafruit_MotorHAT.RELEASE)
            myMotor2.run(Adafruit_MotorHAT.RELEASE)       

    
    #try:
    right_frame = webcam.read()
    left_frame = picam.read()
    
    
 
    # new plan
    left_nav = create_masked_im(left_frame, l_blue_left, u_blue_left)
    right_nav = create_masked_im(right_frame, l_blue_right, u_blue_right)
    
    # distance transform
    full_im = np.concatenate((left_nav, right_nav), axis=1)
    #threading.Thread(target=find_contours, args=(full_im,)).start()

    cv2.imshow("unwrapped", full_im)

    #hsv_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
    #hsv_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HSV)

    # white balance between cameras is inconsistent so it is unlikely that this will work
    # but let's try it anyway
        
    #hue_left, sat_left, val_left = cv2.split(hsv_left)
    #hue_right, sat_right, val_right = cv2.split(hsv_right)

    
    #red_mask_left = cv2.inRange(hsv_left, lower_red_left, upper_red_left)
    #red_mask_right = cv2.inRange(hsv_right, lower_red_right, upper_red_right)

    #homeID_left = cv2.erode(red_mask_left, erode_kernel, iterations=1)

    #homeID_right = cv2.erode(red_mask_right, erode_kernel, iterations=1)        
    
    #left_balance_check = np.sum(np.sum(homeID_left.astype(float), axis=0), axis=0)/500000.0
    #right_balance_check = np.sum(np.sum(homeID_right.astype(float), axis=0), axis=0)/600000.0 - 1.5 # camera tilt bias

    #balance_error = left_balance_check - right_balance_check
        #print(balance_error)
        #print("homing volume")
        #print(left_balance_check)
        #print(right_balance_check)

        # PD control for motor speed
    #vnew_left = des_speed - 5*balance_error
    #vnew_right = des_speed + 5*balance_error

    #if np.abs(balance_error) > 3:
    #        myMotor1.setSpeed(int(vnew_left))
    #        myMotor2.setSpeed(int(vnew_right))
        

    #bchan_l, gchan_l, rchan_l = cv2.split(left_frame)

    #Try this
    # 1. apply hsv-mask, index regions of interest
    # 2. EITHER - apply mask directly to an RGB-channel flow image and see what happens
    # OR split hsv, normalise H, get flow, then apply mask


    #if fcount > 1:

        #if fcount%2 == 0:
        #    threading.Thread(target=findBlockEdge,args=(right_frame,'right',)).start()

        #else:    
            
        #threading.Thread(target=find_edge,args=(left_frame,  l_blue_left, u_blue_left,  1, 640-found_edge_threshold,)).start()
        #threading.Thread(target=find_edge,args=(right_frame, l_blue_right, u_blue_right, 0, found_edge_threshold,)).start()
              
            
      
    #cv2.circle(left_frame, (cX_left,cY_left), 7, (255,0,0), -1)
    #cv2.imshow("Left Camera", left_frame)
    #cv2.imshow("Left Camera", navflow_left)

    #cv2.circle(right_frame, (cX_right,cY_right), 7, (255,0,0), -1)
    #cv2.imshow("Right Camera",  right_frame)
    #cv2.imshow("Right Camera", navflow_right)
        
    
    #bchan_l_prev = bchan_l
    prev_frame = left_frame
    
        
    #homeIDleft_prev = homeID_left
    #homeIDright_prev = homeID_right

    fcount += 1


    #except:
    #    print('no frame')


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
        
        
    if fcount > 100:
        turnOffMotors()
        #running = False
        
webcam.stop()
picam.stop()
cv2.destroyAllWindows()
