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
kernel2 = np.ones((5, 1), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel2, iterations=3)

# Copy the image
imgCopy = cv2.cvtColor(col_images[i].copy(), cv2.COLOR_BGR2RGB)

# Find contours
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Choose the font, scale, color, and thickness
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (0, 255, 0)
thickness = 1

valid_cntrs = []
left_cntrs = []
right_cntrs = []

# Frame dimensions
frame_height, frame_width, _ = imgCopy.shape
divider_position = frame_width // 2  # In this case it's 640 because the frame is 1280 x 720

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (x <= 1280) & (y >= 450) & (cv2.contourArea(contour) >= 4000):
        # Draw the rectangle
        cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the center coordinates of the top edge
        top_center_x = x + w // 2
        top_center_y = y

        # Prepare the text with the top center coordinates
        coordinates_text = f"({top_center_x}, {top_center_y})"

        # Define the position for the text (slightly above the top center of the rectangle)
        text_position = (
            top_center_x - (cv2.getTextSize(coordinates_text, font, font_scale, 1)[0][0] // 2),
            top_center_y - 10)

        # Put the text on the image
        cv2.putText(imgCopy, coordinates_text, text_position, font, font_scale, font_color, 2)

        # Append the contour to valid_cntrs
        valid_cntrs.append(contour)

        # Append to left or right counters based on the x coordinate
        if top_center_x < divider_position:
            left_cntrs.append(contour)
        else:
            right_cntrs.append(contour)

# Count the number of valid contours
num_vehicles = len(valid_cntrs)
num_left_vehicles = len(left_cntrs)
num_right_vehicles = len(right_cntrs)

# Display the number of vehicles detected at the top middle of the frame
vehicle_count_text = f"Vehicles Detected: {num_vehicles}"
text_size, _ = cv2.getTextSize(vehicle_count_text, font, 1, 2)
text_x = (frame_width - text_size[0]) // 2
cv2.putText(imgCopy, vehicle_count_text, (text_x, 40), font, 1, (0, 0, 0), 2)

# Display the number of vehicles on the left half
left_vehicle_text = f"Left: {num_left_vehicles}"
cv2.putText(imgCopy, left_vehicle_text, (60, 100), font, 1, (0, 0, 0), 2)

# Display the number of vehicles on the right half
right_vehicle_text = f"Right: {num_right_vehicles}"
right_text_x = frame_width - 60 - cv2.getTextSize(right_vehicle_text, font, 1, 2)[0][0]
cv2.putText(imgCopy, right_vehicle_text, (right_text_x, 100), font, 1, (0, 0, 0), 2)

# Draw a line
cv2.line(imgCopy, (0, 450), (1280, 450), (0, 255, 255), 1)
plt.imshow(imgCopy)
plt.show()
