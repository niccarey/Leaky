from picamera import PiCamera
from picamera.array import PiRGBArray
from Adafruit_MotorHAT import Adafruit_MotorHAT
import numpy as np 
import threading
import skimage
from skimage.measure import structural_similarity as ssim

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

picam.camera.awb_mode = 'off'
picam.camera.awb_gains=g

# manual white balancing not possible with this driver (?)
# should be ok as long as lighting conditions relatively static
#webcam.stream.set(18,0.5)

# Set up templates and image processing constants

kernsize = 17
temp_size = 40
temp_blur_size = 11
corner_template = np.zeros((240,temp_size))
corner_template[30:240,1:20] = 1
blur_square = np.array(255*cv2.GaussianBlur(corner_template, (kernsize, kernsize),0), dtype=np.uint8)

erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((5,5), np.uint8)

cX_left, cY_left = [0, 0]
cX_right, cY_right = [0, 0]


def turnOffMotors():
    myMotor1.run(Adafruit_MotorHAT.RELEASE)
    myMotor2.run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)    

def findBlockEdge(block_image, location, prev_image):
    global cX_left, cY_left, cX_right, cY_right
    global found_block_flag

    prelim_flow = np.zeros((480,640), np.float)
    hsv_now = cv2.cvtColor(block_image, cv2.COLOR_BGR2HSV)
    hsv_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2HSV)

    bchan_l, gchan_l, rchan_l = cv2.split(block_image)
    bchan_p, gchan_p, rchan_p = cv2.split(prev_image)
    rchan_flow = np.around((bchan_l.astype(np.float) - bchan_p.astype(np.float))/255)
    
    inv_flow = (bchan_p.astype(np.float) - bchan_l.astype(np.float))/255
    #cv2.imshow("reverse flow", inv_flow)
    #cv2.imshow("fwd flow", rchan_flow)
    # need to calculate forward or reverse flow depending on what direction
    # we are travelling
    
    # Convert image to HSV space and threshold
    #hsv = cv2.cvtColor(block_image, cv2.COLOR_BGR2HSV)
    
    if location == 'left':
        lower_blue = np.array([110, 0, 20])
        upper_blue = np.array([150, 100, 255])
        block_locate = cv2.inRange(hsv_now, lower_blue, upper_blue)  
        nav_prev = cv2.inRange(hsv_prev, lower_blue, upper_blue)
        nav_all = block_locate+nav_prev  
        #block_locate = cv2.flip(cv2.transpose(blue_mask),0)

    else:
        lower_blue = np.array([35, 0, 50])
        upper_blue = np.array([110, 255, 255])
        block_detect = cv2.inRange(hsv, lower_blue, upper_blue)
        block_locate = cv2.dilate(block_detect, dilate_kernel, iterations=1)
        
    block_blurred = cv2.GaussianBlur(block_locate, (11,11),0)

    conv_idx = np.where(block_blurred>0)
    
    prelim_flow[conv_idx] = rchan_flow[conv_idx]
    
    cv2.imshow("blue mask", prelim_flow)
        
    # Which is faster: template matching or edge detection?


    # use template as matched filter
    block_u8 = np.array(prelim_flow, dtype=np.uint8)
    template_match = cv2.matchTemplate(block_u8, blur_square, cv2.TM_CCORR_NORMED)

    # Location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    # only update if we're moderately certain about the edge

    if max_val > 0.8:
        found_block_flag = 1
        cX_left = max_loc[0]+temp_size//2 #half template size
        cY_left = max_loc[1]

    else:
        found_block_flag = 0
        


running = True
fcount = 1


des_speed = 140
myMotor1.setSpeed(des_speed)
myMotor2.setSpeed(des_speed)

# note: include motor trim values to compensate for any bias (have to test without tethering)

wheel_count = 1
lower_red_left = np.array([170, 180, 0])
upper_red_left = np.array([180, 255, 255])
lower_red_right = np.array([0, 200, 0])
upper_red_right = np.array([10, 255, 255])


left_balance_check = 0
right_balance_check = 0

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


    hsv_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
    hsv_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HSV)

        # white balance between cameras is inconsistent so it is unlikely that this will work
        # but let's try it anyway
        
    hue_left, sat_left, val_left = cv2.split(hsv_left)
    hue_right, sat_right, val_right = cv2.split(hsv_right)

    
    red_mask_left = cv2.inRange(hsv_left, lower_red_left, upper_red_left)
    red_mask_right = cv2.inRange(hsv_right, lower_red_right, upper_red_right)

    homeID_left = cv2.erode(red_mask_left, erode_kernel, iterations=1)

    homeID_right = cv2.erode(red_mask_right, erode_kernel, iterations=1)        
    
    left_balance_check = np.sum(np.sum(homeID_left.astype(float), axis=0), axis=0)/500000.0
    right_balance_check = np.sum(np.sum(homeID_right.astype(float), axis=0), axis=0)/600000.0 - 1.5 # camera tilt bias

    balance_error = left_balance_check - right_balance_check
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
        

    bchan_l, gchan_l, rchan_l = cv2.split(left_frame)

    #Try this
    # 1. apply hsv-mask, index regions of interest
    # 2. EITHER - apply mask directly to an RGB-channel flow image and see what happens
    # OR split hsv, normalise H, get flow, then apply mask


    if fcount > 1:

        #if fcount%2 == 0:
        #    threading.Thread(target=findBlockEdge,args=(right_frame,'right',)).start()

        #else:        
        threading.Thread(target=findBlockEdge,args=(left_frame,'left', prev_frame,)).start()
        
        
            
        #optic flow not terribly useful?
            #differentiate image differences
            #diff_left = (homeID_left.astype(np.float) - homeIDleft_prev.astype(np.float))/255
            #diff_right = (homeID_right.astype(np.float) - homeIDright_prev.astype(np.float))/255
            #left_flow_check = np.sum(np.sum(diff_left.astype(float), axis=0), axis=0)/100
            #right_flow_check = np.sum(np.sum(diff_right.astype(float), axis=0), axis=0)/100
            #print(left_flow_check)
            #print(right_flow_check)
            #cv2.imshow("Left optic flow", diff_left)
            #cv2.imshow("Right optic flow", diff_right)



    cv2.circle(left_frame, (cX_left,cY_left), 7, (255,0,0), -1)
    cv2.imshow("Left Camera", left_frame)

    #cv2.circle(right_frame, (cX_right,cY_rig  ht), 7, (255,0,0), -1)
    cv2.imshow("Right Camera",  right_frame)
        
    
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
