# import the necessary packages
import cv2
import os
import re
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

fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

for idx, frame in enumerate([i, i+1]):
    axes[idx].imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB))
    axes[idx].set_title("frame: "+str(frame))

plt.tight_layout()  # Adjust layout
plt.show()
