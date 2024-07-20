# Import the necessary packages
import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Frames directory
frames_dir = 'm6_motorway_demo_frames/'
col_frames = os.listdir(frames_dir)

# Sort file names
col_frames.sort(key=lambda f: int(re.sub('\D', '', f)))

# Empty list to store the frames
col_images = []

for i in col_frames:
    # Read the frames
    img = cv2.imread(frames_dir + i)
    # Append the frames to the list
    col_images.append(img)

# Frame number start to compare
i = 20

# Convert the frames to grayscale
grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(col_images[i + 1], cv2.COLOR_BGR2GRAY)

# Differentiate the image
diff_image = cv2.absdiff(grayB, grayA)

# Perform image thresholding
ret, thresh = cv2.threshold(diff_image, 10, 255, cv2.THRESH_BINARY)

# Apply image erosion -> dilation (opening)
kernel = np.ones((2, 2), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=2)
dilated = cv2.dilate(eroded, kernel, iterations=10)

# Draw a line
cv2.line(dilated, (0, 450),(1280,450),(100, 0, 0), 3)

# Plot processed image
plt.imshow(dilated)
plt.show()
