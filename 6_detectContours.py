# import the necessary packages
import cv2
import imutils
import os
import re
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

frames_dir = 'frames/'

col_frames = os.listdir(frames_dir)

# sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# empty list to store the frames
col_images=[]

for i in col_frames:
    # read the frames
    img = cv2.imread(frames_dir+i)
    # append the frames to the list
    col_images.append(img)

i = 15

# convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 50, 255, cv2.THRESH_BINARY)

# apply image erosion and dilation
kernel = np.ones((5,5),np.uint8)
#eroded = cv2.erode(thresh,kernel,iterations = 2)
dilated = cv2.dilate(thresh,kernel,iterations = 1)

# find contours
contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

valid_cntrs = []

for i,cntr in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cntr)
    if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
        valid_cntrs.append(cntr)

# count of discovered contours
len(valid_cntrs)

dmy = col_images[13].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 5)
cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))
plt.imshow(dmy)
plt.show()