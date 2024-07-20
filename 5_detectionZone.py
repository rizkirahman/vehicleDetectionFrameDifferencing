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

i = 13

# convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

diff_image = cv2.absdiff(grayB, grayA)

# perform image thresholding
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

# apply image erosion and dilation
kernel = np.ones((3,3),np.uint8)
eroded = cv2.erode(thresh,kernel,iterations = 2)
dilated = cv2.dilate(eroded,kernel,iterations = 20)

cv2.line(dilated, (0, 600),(1920,600),(100, 0, 0))
plt.imshow(dilated)
plt.show()