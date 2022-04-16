#!/usr/bin/python3

import cv2
import jetson.inference
import jetson.utils

import os
#os.system("apt update")
#os.system("apt install -y fswebcam")
#os.system("pip3 install flask")

import json
import time
import numpy as np

from datetime import datetime
from flask import Flask, render_template, flash, redirect, url_for, request, session,jsonify
from flask import send_file


app = Flask(__name__)

def Camera_Searching():
    camera_id_list = []
    for device in range(0,10):
        stream = cv2.VideoCapture(device)
        
        grabbed = stream.grab()
        stream.release()
        if not grabbed:
            continue
        
        camera_id_list.append(device)
    if camera_id_list:
        print('Available camera IDs are {}'.format(camera_id_list))
        print('Choose the most likely UVC cam id: {}'.format(camera_id_list[0]))
        return camera_id_list[0]
    else:
        print('No available UVC modules!')
        return camera_id_list

def Camera_Searching1():
    stream = cv2.VideoCapture(-1, cv2.CAP_V4L)
    stream.release()

    if stream:
        return stream
    else:
        return camera_id_list

def Camera_Setting(dev_num):
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
    os.system("v4l2-ctl -d /dev/video{} --set-parm={}".format(dev_num,fps))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'brightness',brightness))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'contrast',contrast))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'saturation',saturation))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'white_balance_temperature_auto',white_balance_temperature_auto))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'white_balance_temperature',white_balance_temperature))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'backlight_compensation',backlight_compensation))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'exposure_auto',exposure_auto))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'exposure_absolute',exposure_absolute))
    os.system("v4l2-ctl -d /dev/video{} --set-ctrl={}={}".format(dev_num,'exposure_auto_priority',exposure_auto_priority))
    print('Camera setting finished.')

def uvc_capture(filepath, dev_num):
    ## kindly notice command "/dev/video0" using physical address, do not use "/dev/video0"
    ## Comment below 1 line if default setting for camera is needed
    #Camera_Setting(dev_num)    
    #commands = "fswebcam -d /dev/video{} --no-banner -r 1280x720 {}".format(dev_num, filepath)
    #capture = os.system(commands)
    
    camera = cv2.VideoCapture(dev_num)
    return_value, image = camera.read()
    cv2.imwrite(filepath, image)
    del(camera)

def corefunction(filepath,outputpath, dev_num):
    uvc_capture(filepath, dev_num)

    image_rgb_np =cv2.imread(filepath)
    gray_img = cv2.cvtColor(image_rgb_np, cv2.COLOR_BGR2GRAY)
    h, w = image_rgb_np.shape[:2]
    m = np.reshape(gray_img, [1, w*h])

    if m.sum()/(w*h) > 10:
        img = jetson.utils.loadImage(filepath)
        detections = net.Detect(img, overlay="box,labels,conf")
        jetson.utils.saveImage(outputpath, img)

        count = 0
        for detection in detections:
            if detection.ClassID == 1:
                count+=1
        print('COUNT NUMBER IS------------------------')
        print(count)
        print('------------------------------')
        return count

    else:
        return -1

@app.route('/')
def index():
    a = []
    count_tmp = 0

    while count_tmp < 8:
        inputpath = '/jetson-inference/my-detection-python/imputdata/' + str(count_tmp) + '.jpg'
        outputpath = '/jetson-inference/my-detection-python/outputdata/' + str(count_tmp) +'.jpg'
        ct_tmp = corefunction(inputpath, outputpath, dev_num)

        a.append(ct_tmp)
        count_tmp = count_tmp + 1

        time.sleep(1)
        print('+++++++HUMAN DETECTING LOOP+++++++++++')

    if set((a)) == {-1}:
        return str('Environment is too dark')

    else:
        tmp = np.ceil(np.mean(a))
        print('final count is', int(tmp))
        return str(json.dumps([str(int(tmp)),str(datetime.now())]))

@app.route('/get_image')
def get_image():
    filename = 'outputdata/4.jpg'    
    if os.path.isfile(filename):
        return send_file(filename, mimetype='image/jpg')
    else:
        return str('There is no image file that can be transfered in the target directory')

if __name__ == '__main__':
    dev_num = Camera_Searching()
    net = jetson.inference.detectNet(network="ssd-mobilenet-v2", threshold=0.5)
    Camera_Setting(dev_num)
    # app.run(debug=True)
    app.run(debug=True, port=83, host='172.20.115.28')


