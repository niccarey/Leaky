from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

from PIL import Image
import cv2
import imutils
from imutils.video import VideoStream
from navImage import NavImage

import time
import atexit

erode_kernel = np.ones((7,7), np.uint8)
dilate_kernel = np.ones((7,7), np.uint8)


l_green = np.array([30, 80, 0])
u_green = np.array([80, 255, 255])
l_red = np.array([0, 100, 30])
u_red = np.array([7, 255, 255])

picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start() #this was not the resolution I used previously
time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.1)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain


running = True
fcount = 1

# filters

# set up omnicam image filtering:
cp = [ 389, 297]
r_out = 290;
r_inner = 145;
r_norim = 260;

y,x = np.ogrid[0:600, 0:800]

# Region of interest: occlude non-mirror regions and camera reflection
# (and for some processes, the bottom region of the reflected image)

outer_mask = np.zeros((600,800))
omask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_out**2
outer_mask[omask_px] = 1


inner_mask = np.zeros((600,800))
imask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_inner**2
inner_mask[imask_px] = 1

rim_mask = np.zeros((600,800))
rmask_px = (x-cp[0])**2 + (y-cp[1])**2 <= r_norim**2
rim_mask[rmask_px] = 1

omni_ring = outer_mask - inner_mask
omni_thin = rim_mask - inner_mask


# define polygons for ROI

# front only (may not be used)
poly_front = np.array([cp, [50, 1], [620,1]])
front_mask = np.zeros((600,800))
cv2.fillConvexPoly(front_mask, poly_front, 1)


# back and sides
poly_back = np.array([cp, [1, 600], [800,600], [800,420]])
back_mask = np.zeros((600,800))
cv2.fillConvexPoly(back_mask, poly_back, 1)

poly_left = np.array([cp, [180, 1], [1, 1],[1, 600]])
left_mask = np.zeros((600,800))
cv2.fillConvexPoly(left_mask, poly_left, 1)

poly_right = np.array([cp, [620, 1], [800 ,1], [800, 420]])
right_mask = np.zeros((600,800))
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


depart_flag = 0
deposit_flag = 0
backup_flag = 1

fcount = 1

while running:
    frame = picam.read()
    #key = cv2.waitKey(1) & 0xFF
    
    # need to crop!!! check harvard thing for cropping info (!)
    cropped_frame = np.zeros((600,800,3))
    cropped_frame = frame[196:796, 432:1232,:]
    #cv2.imshow("Check output", cropped_frame)
    if depart_flag:
        # apply mask
        bal_frame = NavImage(cropped_frame)
        bal_frame.convertHsv()
        bal_frame.hsvMask(l_green, u_green)
        
        
        bal_frame.frame[sides_mask < 1] = 0

        bal_frame.erodeMask(erode_kernel, 1)
        bal_frame.dilateMask(dilate_kernel, 1)
        show_frame = bal_frame.frame.copy()
        
        _, cnts, _ = cv2.findContours(bal_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts)>0:
            # find contours with area > (some threshold) (there is a more elegant way of doing this)
            cont_max = max(cnts, key = cv2.contourArea)
            # y is vertical, x is horizontal
            sum_w_x = 0
            sum_w_y = 0
            cnts_lg = [c for c in cnts if cv2.contourArea(c)>1000]
            
            for cnt in cnts_lg:
                M = cv2.moments(cnt)
                cy = int(M['m10']/M['m00']) - cp[0]
                cx = int(M['m01']/M['m00']) - cp[1]
                cent_ang = np.arctan2(cy,cx)
                print("angle mass:", cent_ang*180/np.pi)
                blob_area = cv2.contourArea(cnt)
                # so far, so good
                # dirvec_w[0] is vertical, dirvec_w[1] is horizontal
                dirvec_w = np.array((np.sin(-cent_ang), np.cos(-cent_ang)),dtype = np.float)#*blob_area/cv2.contourArea(cont_max)

                sum_w_y += dirvec_w[0]
                sum_w_x += dirvec_w[1]


            heading_angle = np.arctan2(sum_w_y, sum_w_x)
            headingCOM = np.array((np.sin(heading_angle), np.cos(heading_angle)))*160 + cp
            
            cv2.circle(show_frame, (headingCOM[0].astype(int), headingCOM[1].astype(int)), 7, (255, 255, 255), -1)
            print(heading_angle*180/np.pi)
            #, headingCOM[0], headingCOM[1])

            if (heading_angle > 2.8) or (heading_angle < -2.8) :
                print('walls balanced!')

            elif heading_angle > 0:
                print('left!')

            else:
                print('right!')

        #cv2.imshow("check mask", show_frame)
        img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        imname = './TestIm/CamOutput_'
        imname += str(fcount)
        imname += '.png'
        img.save(imname)

        fcount += 1


    elif deposit_flag:
        
        dep_frame = NavImage(cropped_frame)
        dep_frame.convertHsv()
        dep_frame.hsvMask(l_green, u_green)
        show_frame = dep_frame.frame.copy()
        dep_frame.frame[wide_mask < 1] = 0

        # turn until we have one ROI of a sufficient size, with CoM within a central window
        dep_frame.erodeMask(erode_kernel, 1)
        dep_frame.dilateMask(dilate_kernel, 1)


        _, cnts, _ = cv2.findContours(dep_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # check how many segments larger than (threshold)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        cnts_lg = [c for c in cnts if cv2.contourArea(c)>1000]
        # calculate centroids (?)
        # mnts = [cv2.moments(c) for c in cnts_lg]
        if len(cnts_lg) > 0:
            if len(cnts_lg) > 1:
                print("Keep turning")

            else: # we have one large blob, where is it?
                blob = cnts_lg[0]
                M = cv2.moments(blob)

                cy = int(M['m10']/M['m00']) - cp[0]
                cx = int(M['m01']/M['m00']) - cp[1]
                cent_ang = np.arctan2(cy,cx)
                
                print(cent_ang*180/np.pi)

                if (cent_ang > 2.8) or (cent_ang < -2.8):
                    print("heading straight")
                elif cent_ang < 0:
                    print("head right")
                else:
                    print("head  left")


        #cv2.imshow("check mask", show_frame)

    elif backup_flag:
        
        
        back_frame = NavImage(cropped_frame)
        back_frame.convertHsv()
        back_frame.hsvMask(l_red, u_red)
        back_frame.frame[wide_mask < 1] = 0


        # turn until we can see two contours in roughly the right position
        back_frame.erodeMask(erode_kernel,1)
        back_frame.dilateMask(dilate_kernel, 1)
        show_frame = back_frame.frame.copy()

        _, cnts, _ = cv2.findContours(back_frame.frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            sum_x = 0
            sum_y = 0
            cnts_lg = [c for c in cnts if cv2.contourArea(c)>200]
    
            if len(cnts_lg) < 2:
                print("keep turning")

            else:
                cnt_sort = sorted(cnts_lg, key=cv2.contourArea, reverse=True)
                MoTs = [cv2.moments(cnt_sort[0]), cv2.moments(cnt_sort[1])]
            
                for M in MoTs:
                    cy = int(M['m10']/M['m00']) - cp[0]
                    cx = int(M['m01']/M['m00']) - cp[1]
                    cent_ang = np.arctan2(cy,cx)
                
                    dirvec = [np.sin(cent_ang), np.cos(cent_ang)]
                    sum_y += dirvec[0]
                    sum_x += dirvec[1]
            

                heading_angle = np.arctan2(sum_y, sum_x)

                print("home spotted! Steer as normal", heading_angle*180/np.pi)

        #cv2.imshow("Mask testing", show_frame)
        img = Image.fromarray(show_frame)
        imname = './TestIm/BackupTest_'
        imname += str(fcount)
        imname += '.png'
        img.save(imname)
        fcount += 1


    #if key == ord("q"):
    #    running=False
    #    break


