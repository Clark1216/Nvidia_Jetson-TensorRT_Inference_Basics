#!/usr/bin/python3

'''
This script does the following:
1. process and store a single frame
2. returns number of people in single frame
'''

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

def corefunction(filepath,outputpath):
    uvc_capture(filepath)
    # input = jetson.utils.videoSource(filepath)
    #output = jetson.utils.videoOutput(outputpath)
    net = jetson.inference.detectNet(network="ssd-mobilenet-v2", threshold=0.5)

    # img = input.Capture()
    img = jetson.utils.loadImage(filepath)
    detections = net.Detect(img, overlay="box,labels,conf")
    #output.Render(img)
    #output.Close()
    jetson.utils.saveImage(outputpath, img)

    count = 0
    for detection in detections:
        if detection.ClassID == 1:
            count+=1

    print("Number of persons in frame is " + str(count))

filepath = "input.jpg"
outputpath = "output.jpg"
Camera_Setting()
corefunction(filepath, outputpath)


