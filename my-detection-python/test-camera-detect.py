#!/usr/bin/python3

import cv2

def Camera_Searching():
    camera_id_list = []
    for device in range(0,10):
        stream = cv2.VideoCapture(device)
        
        grabbed = stream.grab()
        stream.release()
        stream.destroyAllWindows()
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

a = Camera_Searching()
print(a)
