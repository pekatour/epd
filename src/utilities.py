import numpy as np
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt

# Function to load an image from a file path into a numpy array
# IN : file_path (str) - Path to the image file
#      gray (bool) - If True, load the image in grayscale
#      verbose (bool) - If True, print image information and show the image
# OUT : image (ndarray) - Loaded image as a numpy array
def load_image(file_path, gray=False, verbose=False):
    image = io.imread(file_path, as_gray=gray)
    if verbose:
        print(type(image))
        print("Image shape:", image.shape)
        print("Data type:", image.dtype)

        io.imshow(image)
        io.show()
    return image