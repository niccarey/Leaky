import cv2
import numpy as np

class NavImage(object):
    # This is a bit fast and loose as the NavImage object can change from 
    # a 3-dimensional to a 1 dimensional array without warning.
    # Need to make this a bit safer.
    
    def __init__(self, frame):
        self.frame = frame
    
    
    def convertHsv(self):
        hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.frame = hsv_frame
        
    def hsvMask(self, lower_bound, upper_bound):
        mask = cv2.inRange(self.frame, lower_bound, upper_bound)
        self.frame = mask
        
    def erodeMask(self, erode_kernel, iteration):
        mask = cv2.erode(self.frame, erode_kernel, iterations=iteration)
        self.frame = mask
        
    def dilateMask(self, dilate_kernel, iteration):
        mask = cv2.dilate(self.frame, dilate_kernel, iterations=iteration)
        self.frame = mask
        
    def maskWeight(self, norm_val, offset_val):
        maskedIm = self.frame
        weight = np.sum(np.sum(maskedIm.astype(float), axis=0), axis=0)/norm_val - offset_val
        return weight
        
        
