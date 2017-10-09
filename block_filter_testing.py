from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np 
import threading

import imutils
from imutils.video import VideoStream
import cv2
import time
import atexit
import os


# initialise pycam
g = (1.5,2.1)

# Initialize webcam

webcam = VideoStream(src=0).start()
picam = VideoStream(usePiCamera=True, resolution=(640,480)).start()

time.sleep(0.5)

picam.camera.awb_mode = 'off'
picam.camera.awb_gains=g

os.system('v4l2-ctl --set-ctrl=white_balance_temperature_auto=0')
os.system('v4l2-ctl --set-ctrl=white_balance_temperature=2800')
os.system('v4l2-ctl --set-ctrl=exposure_auto=1')
os.system('v4l2-ctl --set-ctrl=exposure_absolute=200')
os.system('v4l2-ctl --set-ctrl=brightness=0')       


# Set up templates and image processing constants
# Matched filtering:
kernsize = 17
temp_size = 40
temp_blur_size = 11
corner_template = np.zeros((240,temp_size))
corner_template[30:240,1:20] = 1

left_template = np.array(255*cv2.GaussianBlur(corner_template, (kernsize, kernsize),0), dtype=np.uint8)
right_template = cv2.flip(left_template, 1)

cX_left, cY_left = [0, 0]
cX_right, cY_right = [0, 0]

erode_kernel = np.ones((5,5), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)

l_red_left = np.array([170, 180, 0])
u_red_left = np.array([180, 255, 255])
l_red_right = np.array([0, 200, 0])
u_red_right = np.array([10, 255, 255])

l_blue_left = np.array([110, 0, 20])
u_blue_left = np.array([150, 100, 255])
l_blue_right = np.array([100, 0, 0])
u_blue_right = np.array([150, 100, 255])



def findBlockEdge(block_image, location):
    global cX_left, cY_left, cX_right, cY_right

    # Convert image to HSV space and threshold
    hsv = cv2.cvtColor(block_image, cv2.COLOR_BGR2HSV)
    
    if location == 'left':
        block_locate = cv2.inRange(hsv, l_blue_left, u_blue_left)    
        block_blurred = cv2.GaussianBlur(block_locate, (11,11),0)
        block_u8 = np.array(block_blurred, dtype=np.uint8)
        template_match = cv2.matchTemplate(block_u8, left_template, cv2.TM_CCORR_NORMED)


    else:
        block_detect = cv2.inRange(hsv, l_blue_right, u_blue_right)
        block_locate = cv2.dilate(block_detect, dilate_kernel, iterations=1)
        block_blurred = cv2.GaussianBlur(block_locate, (11,11),0)
        block_u8 = np.array(block_blurred, dtype=np.uint8)
        template_match = cv2.matchTemplate(block_u8, right_template, cv2.TM_CCORR_NORMED)

    # Location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)

    # only update if we're moderately certain about the edge

    if max_val > 0.8:
        if location == 'left':
            cX_left = max_loc[0]+temp_size//2 #half template size
            cY_left = max_loc[1]
            print("Left block edge: ", cX_left)

        else:
            cX_right = max_loc[0] +temp_size//2
            cY_right = max_loc[1]
            print("Right block edge: ", cX_right)




running = True
fcount = 1


while running:

    # two problems: slow image processing (can thread)
    # second: how do we tell when to stop?
    
    try:
        right_frame = webcam.read()
        left_frame = picam.read()


        #hsv_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)
        #hsv_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2HSV)

        # white balance between cameras is inconsistent so it is unlikely that this will work
        # but let's try it anyway
        
        #hue_left, sat_left, val_left = cv2.split(hsv_left)
        #hue_right, sat_right, val_right = cv2.split(hsv_right)
    
        #red_mask_left = cv2.inRange(hsv_left, l_red_left, u_red_left)
        #red_mask_right = cv2.inRange(hsv_right, l_red_right, u_red_right)
        #homeID_left = cv2.erode(red_mask_left, erode_kernel, iterations=1)
        #homeID_right = cv2.erode(red_mask_right, erode_kernel, iterations=1)        
        if fcount%2 == 0:
            threading.Thread(target=findBlockEdge,args=(right_frame,'right',)).start()

        else:        
            threading.Thread(target=findBlockEdge,args=(left_frame,'left',)).start()
        
        cv2.circle(left_frame, (cX_left,cY_left), 7, (255,0,0), -1)
        cv2.imshow("Left Camera",  left_frame)


        cv2.circle(right_frame, (cX_right,cY_right), 7, (255,0,0), -1)
        cv2.imshow("Right Camera",  right_frame)
        

        fcount += 1


    except:
        print('no frame')


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
        
        
        
webcam.stop()
picam.stop()
cv2.destroyAllWindows()




