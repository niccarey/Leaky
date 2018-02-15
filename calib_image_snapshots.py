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


picam = VideoStream(usePiCamera=True, resolution=(1648,1232)).start()
time.sleep(0.5)

print("Setting camera gains and white balance ")
camgain = (1.4,2.1)
picam.camera.awb_mode = 'off'
picam.camera.awb_gains = camgain


running = True
fcount = 1


while running:
    frame = picam.read()
    key = cv2.waitKey(1) & 0xFF
    
    cv2.imshow("Visualisation", frame)
    
        
    if key == ord("a"):
        img = Image.fromarray(frame)
        imname = './CalibIm/OmniCal_'
        imname += str(fcount)
        imname += '.png'
        img.save(imname)
        fcount += 1
		
    elif key == ord("q"):
        running=False
        break


