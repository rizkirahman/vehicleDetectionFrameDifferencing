# Import the necessary packages
import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Frames directory
frames_dir = 'm6_motorway_demo_frames/'
col_frames = os.listdir(frames_dir)

# Filter out files that do not contain any digits
col_frames = [f for f in col_frames if re.search('\d', f)]

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

# Apply morphological transformations
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel2, iterations=3)

# Plot processed image
plt.imshow(dilated, cmap='gray')
plt.show()
