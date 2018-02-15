import cv2
import numpy as np
import os

from leaky_nav_functions import *
from navImage import NavImage

checkim = cv2.imread('check_homing_3.png')
l_check, u_check = boundary_estimate(checkim, 90,120, 100, 255, 80, 255, 15)

imwhat = NavImage(checkim)
imwhat.convertHsv()
imwhat.hsvMask(l_check, u_check)

cv2.imshow("original", checkim)
cv2.waitKey(0)
cv2.imshow("masked", imwhat.frame)
cv2.waitKey(0)
