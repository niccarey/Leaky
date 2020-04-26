import numpy as np
import cv2

class KeypointExpanded(object):
    
    def __init__(self, kps):
        self.keypoints = kps
        self.heights = []
        
    def add_height(self, new_height):
        height_vec = self.heights
        height_vec.extend(new_height)
        self.heights = height_vec
        
