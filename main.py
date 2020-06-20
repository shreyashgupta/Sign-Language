# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:13:49 2020

@author: Shreyash
"""
import numpy as np
import cv2
abc=0
from aslModel import ASLModel
import math
col=(0,0,0)
import statistics
# Open Camera
capture = cv2.VideoCapture(0)
l=0
buffer = [None] * 3
model=ASLModel("model1.json", "model_weights1.h5")
while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()
    frame_copy=frame.copy()
    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 100), (300, 300),col,1)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
       
        w2=max(h,w)

        a=int(w2/2)        
        cx=x+int(w/2)
        cy=y+int(h/2)
        crop3=frame_copy[95+cy-a:105+cy+a,95+cx-a:105+cx+a]
        crop3= cv2.cvtColor(crop3, cv2.COLOR_BGR2GRAY)
        _,crop3=cv2.threshold(crop3,127,255, cv2.THRESH_BINARY_INV)
        cv2.imshow('crop3',crop3)
        
        crop3 = cv2.resize(crop3, (256,256), interpolation = cv2.INTER_AREA)
        final1 = np.resize(crop3, (256,256, 1))
        final = np.expand_dims(final1, axis=0)
        res=model.predict(final)
        buffer[l]=res
        l=l+1
        if(l%3==0):
            l=0
            abc=statistics.mode(buffer)
            buffer=[None] * 3
        cv2.rectangle(frame,(100,75),(300,100),col,-1)
        cv2.putText(frame,str(abc), (190, 95), cv2.FONT_HERSHEY_COMPLEX, .75, (0,255,0), lineType=cv2.LINE_AA)
        # Find convex hull
        hull = cv2.convexHull(contour)
        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv2.line(crop_image, start, end, [0, 255, 0], 2)

    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

