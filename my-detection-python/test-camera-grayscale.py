#!/usr/bin/python3

'''
This script does the following:
1. measure gayscale to determine if image captured is too dark
'''

import numpy as np
import cv2
import jetson.inference
import jetson.utils

import os

def Camera_Setting():
    #setting attributes
    fps = "15"
    brightness = "64"
    contrast = "12"
    saturation = "55"
    white_balance_temperature_auto = "1"
    white_balance_temperature = "5000"
    backlight_compensation = "3"
    exposure_auto = "3"
    exposure_absolute = "179"
    exposure_auto_priority = "0"
    #setting commands
    os.system("v4l2-ctl -d /dev/video0 --set-parm={}".format(fps))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('brightness',brightness))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('contrast',contrast))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('saturation',saturation))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('white_balance_temperature_auto',white_balance_temperature_auto))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('white_balance_temperature',white_balance_temperature))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('backlight_compensation',backlight_compensation))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('exposure_auto',exposure_auto))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('exposure_absolute',exposure_absolute))
    os.system("v4l2-ctl -d /dev/video0 --set-ctrl={}={}".format('exposure_auto_priority',exposure_auto_priority))
    print('Camera setting finished.')

def uvc_capture(filepath):
    #kindly notice command "/dev/video0" using physical address, do not use "/dev/video0" 
    commands = "fswebcam -d /dev/video0 --no-banner -r 1280x720 {}".format(filepath)
    capture = os.system(commands)

if __name__ == '__main__':

    filepath='imputdata/cam_test.jpg'

    uvc_capture(filepath)
    image_rgb_np =cv2.imread(filepath)
    gray_img = cv2.cvtColor(image_rgb_np, cv2.COLOR_BGR2GRAY)
    h, w = image_rgb_np.shape[:2]
    m = np.reshape(gray_img, [1, w*h])
    print("mean value of greyscale",m.sum()/(w*h))


