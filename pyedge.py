"""
pyEdge: a small utility to detect edges in images.
It supports three different methods for edge detection: Canny, Sobel and Prewitt
"""

# Import libraries
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny

# TODO：these should be passed by the user
# if no output filename is passed, it should be generated automatically
input_filename = "test_images/DAPI.png"
output_filename = "test_images/DAPI_edges.png"

# Read image and find edges
img = imread(input_filename)
# TODO: user should choose method
img_edges = canny(img, sigma=7)

# Save to file
imsave(fname=output_filename, arr=img_edges)

# Display images
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray")
ax[1].imshow(img_edges, cmap="gray")

for a in ax:
    a.axis("off")

plt.show()