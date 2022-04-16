from flask import Flask, render_template, flash, redirect, url_for, request, session,jsonify
#import requests
from time import sleep

from PIL import Image
from datetime import datetime

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw, ImageFont
import time
import six.moves.urllib as urllib
import tarfile
import io


import os
import sys
import re
sys.path.append('/tensorflow/models/research')
sys.path.append('/tensorflow/models/research/slim')
sys.path.append('/tensorflow/models/research/object_detection')
from object_detection.utils import visualization_utils as viz_utils

from datetime import datetime
from PIL import Image


import requests
import json
from six import BytesIO

# from picamera.array import PiRGBArray
# from picamera import PiCamera

from threading import Timer

#BSD License need to declare for picamera,https://opensource.org/licenses/BSD-3-Clause
app = Flask(__name__)

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

def downloadModel(MODEL_URL):
    firstpos = MODEL_URL.rfind("/")
    lastpos = MODEL_URL.rfind(".")
    MODEL_NAME = MODEL_URL[firstpos + 1:lastpos]
    MODEL_FILE = MODEL_NAME + '.tar.gz'

    print("Preparing to download tensorflow model {}".format(MODEL_FILE))
    print("Is file downloaded? %s " % os.path.isfile(MODEL_FILE))
    if not os.path.isfile(MODEL_FILE):
        opener = urllib.request.URLopener()
        opener.retrieve(MODEL_URL, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            tar_file.extract(file, os.getcwd())

def uvc_capture(filepath):
    #kindly notice command "/dev/video0" using physical address, do not use "/dev/video0" 
    commands = "fswebcam -d /dev/video0 --no-banner -r 1280x720 {}".format(filepath)
    capture = os.system(commands)

def loadTensorflowModel():
    start_time = time.time()
    tf.keras.backend.clear_session()
    detect_fn = tf.saved_model.load(
        '/tensorflow/models/research/object_detection/test_data/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time: ' + str(elapsed_time) + 's')
    return detect_fn

def doinference(image_np):
    input_tensor = np.expand_dims(image_np, 0)
    start_time = time.time()
    detections = detect_fn(input_tensor)
    end_time = time.time()
    elapsed.append(end_time - start_time)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(np.int32),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.40,
        agnostic_mode=False)

    return [image_np_with_detections,detections['detection_classes'][0].numpy().astype(np.int32), detections['detection_scores'][0].numpy()]

def corefunction(filepath,outputpath):
    
#     filepath='/app/imputdata/1.jpg'
#     outputpath="/app/outputdata/1.jpg"
    print("Camera starting")
    uvc_capture(filepath)
    image_rgb_np =cv2.imread(filepath)
    #frame = piFrame.array
    #image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(image_rgb_np, cv2.COLOR_BGR2GRAY)
    h, w = image_rgb_np.shape[:2]
    m = np.reshape(gray_img, [1, w*h])
    if m.sum()/(w*h) > 10:
        image_rgb_np_with_detections, category_index, score= doinference(image_rgb_np)
        image_bgr_np_with_detections = cv2.cvtColor(image_rgb_np_with_detections, cv2.COLOR_RGB2BGR)
        print("saving output image")
        cv2.imwrite(outputpath,image_bgr_np_with_detections)
        count=0
        for i in range(len(score)):
            if score[i]>0.45 and category_index[i]==1:
                count=count+1
    #             print('-----------------')
    #             print('TOTAL HUMAN NUMBER IS')
    #             print(i, end=" ")
    #             print(' ')
    #             print('-----------------')
    #     print(category_index)
    #     print(score)
        print('COUNT NUMBER IS------------------------')
        print(count)
        print('------------------------------')
        return count
    else:
        print('Pic is too dark, gery value is only {}'.format(m.sum()/(w*h)))
        return -1

################################
###  Function Def:        ######
###   Preprocess & Decode QR ###
################################
def decoder(threshold,file_path1,file_path2):#Input is the binarization threshold

    #Path below is only for testing and debugging
    file_path_static = '/home/pi/Desktop/DuDu_Bench/Successful_Sample2_A0169.png'
        
#######################
###  Preprocessing ####
#######################
    img_tmp = Image.open(file_path1)
    img_tmp_BL = img_tmp.convert('L')#Convert to gray img

    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
            
    #Binarize the image
    img_tmp_BIN = img_tmp_BL.point(table,'1')
    img_tmp_BIN.save(file_path2)

@app.route('/')
def index():
    a= []
#     def sleeptime(hour,min,sec):
#         return hour*3600 + min*60 + sec;
#     second = sleeptime(0,0,2);
    count_tmp = 0
    while count_tmp < 8:
        #     filepath='/app/imputdata/1.jpg'
#     outputpath="/app/outputdata/1.jpg"
        inputpath = os.path.join('/app/imputdata/',str(count_tmp)+'.jpg')
        outpath = os.path.join('/app/outputdata/',str(count_tmp)+'.jpg')
        ct_tmp = corefunction(inputpath, outpath)
        if ct_tmp != -1:
            a.append(ct_tmp)
            count_tmp = count_tmp + 1
#         time.sleep(second);
        print('+++++++HUMAN DETECTING LOOP+++++++++++')
    if a:
        tmp = np.ceil(np.mean(a))
        print('final count is', int(tmp))
        #return str(int(tmp))
        return str(json.dumps([str(int(tmp)),str(datetime.now())]))
    else:
        return str('Looping is spoiled, pls contact A*STAR developer Dr. Du Pengfei')

if __name__ == '__main__':
    print("loading label maps")
    # Load the COCO Label Map
    elapsed = []
    category_index = {
        1: {'id': 1, 'name': 'person'},
        2: {'id': 2, 'name': 'bicycle'},
        3: {'id': 3, 'name': 'car'},
        4: {'id': 4, 'name': 'motorcycle'},
        5: {'id': 5, 'name': 'airplane'},
        6: {'id': 6, 'name': 'bus'},
        7: {'id': 7, 'name': 'train'},
        8: {'id': 8, 'name': 'truck'},
        9: {'id': 9, 'name': 'boat'},
        10: {'id': 10, 'name': 'traffic light'},
        11: {'id': 11, 'name': 'fire hydrant'},
        13: {'id': 13, 'name': 'stop sign'},
        14: {'id': 14, 'name': 'parking meter'},
        15: {'id': 15, 'name': 'bench'},
        16: {'id': 16, 'name': 'bird'},
        17: {'id': 17, 'name': 'cat'},
        18: {'id': 18, 'name': 'dog'},
        19: {'id': 19, 'name': 'horse'},
        20: {'id': 20, 'name': 'sheep'},
        21: {'id': 21, 'name': 'cow'},
        22: {'id': 22, 'name': 'elephant'},
        23: {'id': 23, 'name': 'bear'},
        24: {'id': 24, 'name': 'zebra'},
        25: {'id': 25, 'name': 'giraffe'},
        27: {'id': 27, 'name': 'backpack'},
        28: {'id': 28, 'name': 'umbrella'},
        31: {'id': 31, 'name': 'handbag'},
        32: {'id': 32, 'name': 'tie'},
        33: {'id': 33, 'name': 'suitcase'},
        34: {'id': 34, 'name': 'frisbee'},
        35: {'id': 35, 'name': 'skis'},
        36: {'id': 36, 'name': 'snowboard'},
        37: {'id': 37, 'name': 'sports ball'},
        38: {'id': 38, 'name': 'kite'},
        39: {'id': 39, 'name': 'baseball bat'},
        40: {'id': 40, 'name': 'baseball glove'},
        41: {'id': 41, 'name': 'skateboard'},
        42: {'id': 42, 'name': 'surfboard'},
        43: {'id': 43, 'name': 'tennis racket'},
        44: {'id': 44, 'name': 'bottle'},
        46: {'id': 46, 'name': 'wine glass'},
        47: {'id': 47, 'name': 'cup'},
        48: {'id': 48, 'name': 'fork'},
        49: {'id': 49, 'name': 'knife'},
        50: {'id': 50, 'name': 'spoon'},
        51: {'id': 51, 'name': 'bowl'},
        52: {'id': 52, 'name': 'banana'},
        53: {'id': 53, 'name': 'apple'},
        54: {'id': 54, 'name': 'sandwich'},
        55: {'id': 55, 'name': 'orange'},
        56: {'id': 56, 'name': 'broccoli'},
        57: {'id': 57, 'name': 'carrot'},
        58: {'id': 58, 'name': 'hot dog'},
        59: {'id': 59, 'name': 'pizza'},
        60: {'id': 60, 'name': 'donut'},
        61: {'id': 61, 'name': 'cake'},
        62: {'id': 62, 'name': 'chair'},
        63: {'id': 63, 'name': 'couch'},
        64: {'id': 64, 'name': 'potted plant'},
        65: {'id': 65, 'name': 'bed'},
        67: {'id': 67, 'name': 'dining table'},
        70: {'id': 70, 'name': 'toilet'},
        72: {'id': 72, 'name': 'tv'},
        73: {'id': 73, 'name': 'laptop'},
        74: {'id': 74, 'name': 'mouse'},
        75: {'id': 75, 'name': 'remote'},
        76: {'id': 76, 'name': 'keyboard'},
        77: {'id': 77, 'name': 'cell phone'},
        78: {'id': 78, 'name': 'microwave'},
        79: {'id': 79, 'name': 'oven'},
        80: {'id': 80, 'name': 'toaster'},
        81: {'id': 81, 'name': 'sink'},
        82: {'id': 82, 'name': 'refrigerator'},
        84: {'id': 84, 'name': 'book'},
        85: {'id': 85, 'name': 'clock'},
        86: {'id': 86, 'name': 'vase'},
        87: {'id': 87, 'name': 'scissors'},
        88: {'id': 88, 'name': 'teddy bear'},
        89: {'id': 89, 'name': 'hair drier'},
        90: {'id': 90, 'name': 'toothbrush'},
    }
    # start of main code
    # initialize the TF and relavant model
    MODEL_URL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    os.chdir("/tensorflow/models/research/object_detection/test_data/")
    downloadModel(MODEL_URL)
    print("loading DL model")
    detect_fn= loadTensorflowModel()
    print("Model Loaded, app running")
    os.chdir("/app")
    app.run(debug=True, port=83, host='127.0.0.1')
    # app.run(debug=True, port=83, host='172.20.115.125')

