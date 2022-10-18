"""
pyEdge: a small utility to detect edges in images.
It supports three different methods for edge detection: Canny, Sobel and Prewitt
"""

# Import libraries
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import sobel, prewitt
from skimage.feature import canny
import numpy as np
import argparse
import os

# We create a parser to read command line arguments
# We will call our software as python.pyedge.py -- imput image.png [-- output image_edges.png] [-- method canny] [-- cmap virdis]
# Default for method is canny and for cmap is gray
# If no output is parsed we will name the output as xxxx_edges.png
parser = argparse.ArgumentParser(description="Detect edges in an image")
parser.add_argument("-i", "--input", help="The input image file", required=True)
parser.add_argument("-o", "--output", help="The output image file", required=False)
parser.add_argument("-m", "--method", help="The edge detecting method, default is canny", default ="canny", required=False, 
                   choices=["canny", "sobel", "prewitt"])
parser.add_argument("-c", "--colourmap", help="The colour map, default is gray", default="gray", required=False)

args = parser.parse_args()

# Functions 
def find_edges(img:np.array, method:str="canny") -> np.array:
    """Find edges in an image, using the method chosen by the user (default is canny)

    Args:
        img (np.array): The input image
        method (str, optional): The edge-detection method. Defaults to "canny".
                                Allowed values are "canny", "prewitt", "sobel".

    Returns:
        np.array: The detected edges
    """
    
    if method == "canny":
        img_edges = canny(img, sigma=7)
    elif method == "sobel":
        img_edges = sobel(img)
    elif method == "prewitt":
        img_edges = prewitt(img)
    else:
        raise ValueError(f"{method} is not a recognised method.")

    return img_edges

def display_images(img:np.array, img_edges: np.array, cmap:str="gray")-> None:
    """Displays the image and the edges side by side

    Args:
        img (np.array): The input image
        img_edges(np.array): The image of the edges
        cmap (str): The colormap. Default is gray
    
    Returns: None -  displays the images
    """

    # Display images
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    ax[0].imshow(img, cmap=cmap)
    ax[1].imshow(img_edges, cmap=cmap)
    
    for a in ax:
        a.axis("off")
    
    plt.show()

# if no output filename is passed, it should be generated automatically
# Check that the file exists
if os.path.exists(args.input):
    input_filename = args.input
else:
    raise FileExistsError(f"The input file {args.input} dose not exsit. Exiting pyEdge.")

if args.output is None:
    # Split the input_filename on the dot image.jpg -> ("image", "jpg")
    input_split = input_filename.split(".")
    output_filename = f"{input_split[0]}_edges.{input_split[1]}"
    print("No output filename was given, saving to {output_filename}")
else:
    output_filename = args.output

# Read image and find edges
img = imread(input_filename)

img_edges = find_edges(img, args.method)

# Save to file
imsave(fname=output_filename, arr=img_edges)

display_images(img, img_edges, args.colourmap)