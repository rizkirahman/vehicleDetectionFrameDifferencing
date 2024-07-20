# Import the necessary packages
import cv2
import os
import re
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

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for idx, frame in enumerate([i, i + 1]):
    axes[idx].imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    axes[idx].set_title("Frame: " + str(frame))

# Adjust layout
plt.tight_layout()
plt.show()
