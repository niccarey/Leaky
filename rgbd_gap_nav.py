#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from pythonds.basic.stack import Stack
from scipy import ndimage
from scipy.ndimage import morphology as scmorph
from collections import deque

import logging
import time
import numpy as np
import cv2
import pyrealsense as pyrs
import atexit
import _thread as thread

# set up logging level
logging.basicConfig(level=logging.INFO)


# -- INITIALISING

# global variables or anything that needs to be initialised outside a function goes here

# set filter coefficients (currently first order filter)
c0 = 1
c1 = 0.5
thresh = 20

# initialise constants used in navigation loop
yaw_error = 0
est_dist = 100

# other variables to pass between threads:
cX, cY = [0, 0]
mean_disp = 0


def depthmap_flow_nav(d_im):
    # calculates a crude depth flow field and identifies a likely clear path
    # by template matching to a gaussian distance function
    global cX, cY
    global yaw_error
    global est_dist

    # create nxn zeros and appropriate kernel
    kernlen = 321
    dirac_im = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    dirac_im[kernlen // 2, kernlen // 2] = 1

    # gaussian-smooth the dirac, resulting in a gaussian filter:
    gauss_template = cv2.GaussianBlur(dirac_im, (kernlen, kernlen), 0)
    # normalise
    max_g = max(gauss_template.max(axis=1))
    gauss_display = np.array(255 * gauss_template / max_g, dtype=np.uint8)

    # filter the distance output to remove discontinuities and approximate a flow field
    d_im_filt = scmorph.grey_closing(d_im, size=(7, 7))
    blur = cv2.GaussianBlur(d_im_filt, (71, 71), 0)

    # we may want to restrict our analysis to a central 'band' of the image
    # can use a mask in the template match for this
    blur = np.array(blur, dtype=np.uint8)

    # Cross correlate a gaussian peaked function of size < image:
    template_match = cv2.matchTemplate(blur, gauss_display, cv2.TM_CCORR_NORMED)

    template_match = cv2.normalize(template_match, 0, 1, cv2.NORM_MINMAX)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match)
    # max_loc gives top left of template match
    cX = max_loc[0] + kernlen // 2
    cY = max_loc[1] + kernlen // 2

    # distance: examine distance at max_loc to estimate whether we have hit a wall? This doesn't work very well!
    vis_cent = blur[(cX - 8):(cX + 8), (cY - 8):(cY + 8)]
    vis_cent = vis_cent.astype('float64')
    vis_cent[vis_cent < 5] = np.nan
    est_dist = np.nanmean(vis_cent)

    yaw_error = cX - 640 / 2  # change this for different depth image size


# -- MAIN CONTROL LOOP
if __name__ == '__main__':
    with pyrs.Service() as pyrs:
        # wee bit of overkill in our main definition here
        motorrunning = True

        dev = pyrs.Device()
        dev.set_device_option(29, 2)

        frameint = 5
        framecount = 0
        fstart = 1
        r_ch_prev = 0
        b_ch_prev = 0
        g_ch_prev = 0

        while motorrunning:

            # Get colour-aligned-depth and depth frames (or vice versa!)
            # if framecount > 1
            # separate into channels
            # subtract previous colour channels from current
            # scale to (0,1)
            # view differential channels
            # scale depth to (0,1)
            # view depth


            timethen = time.time()
            framecount += 1

            ## IMAGE PROCESSING
            dev.wait_for_frames()
            c_im = dev.color
            rgb_im = c_im[..., ::-1]

            # scaling to map better to color/grayscale
            d_im = dev.dac*0.0002
            d_inv = 1-d_im
            d_inv[d_inv>0.96] = 0

            d_im_col = cv2.applyColorMap(d_inv.astype(np.uint8), cv2.COLORMAP_HOT)
            # I don't like how noisy the depth image is.
            d_im_filt = scmorph.grey_closing(d_inv, size=(7, 7))
            blur = cv2.GaussianBlur(d_im_filt, (11, 11), 0)

            # normalize (sort of) so optimal range is roughly 0->1
            #d_scaled = d_im.astype(np.float)*0.1

            # set background to zero
            # invert to correspond better with optic flow map
            #d_inv = 1 - d_scaled

            #d_bg = d_scaled
            #d_bg = 255 - d_bg
            #d_bg[d_bg < 0] = 0
            d_bg = blur

            # Separate into channels
            b_ch, g_ch, r_ch = cv2.split(rgb_im)

            if fstart > 1:
                # scaled image difference
                r_diff = (r_ch.astype(np.float) - r_ch_prev.astype(np.float))/255
                conv_idx = (blur < 0.01)
                d_bg[conv_idx] = r_diff[conv_idx]

                rgb_and_depth = np.concatenate((d_bg, d_im), axis=1)
                cv2.imshow('', rgb_and_depth)

            # every nth frame, update direction by analysing depth map
            # error sampling rate << framerate, to try and reduce jitter. Could also sample at a higher rate and filter.

            # two nav options: segmentation or gradient. Segmentation is a problem with very noisy images
            if framecount > frameint:
                yaw_e_prev = yaw_error
                # thread.start_new_thread(depthmap_seg_nav, (d_im_col, ))
                thread.start_new_thread(depthmap_flow_nav, (d_im,))
                framecount = 1

            cv2.circle(d_im_col, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(d_im_col, str(yaw_error), (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # cd = np.concatenate((blur, gauss_template), axis=1)
            cd = np.concatenate((rgb_im, d_im_col), axis=1)

            # I do not recommend using local display over wifi unless you don't care if the robot runs in circles
            # but if connected to ethernet, you can change REMOTE_VIEW flag and uncomment the below
            #cv2.imshow('', cd)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                motorrunning=False

            timenow = time.time()
            # print(timenow - timethen)
            b_ch_prev = b_ch
            g_ch_prev = g_ch
            r_ch_prev = r_ch

            fstart += 1


            # time.sleep(1)
        print("ended, stopping all motors")
