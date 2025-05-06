from skimage import io
import numpy as np
import matplotlib.pyplot as plt

# Function to load an image from a file path into a numpy array
# IN : file_path (str) - Path to the image file
#      gray (bool) - If True, load the image in grayscale
#      verbose (bool) - If True, print image information and show the image
# OUT : image (ndarray) - Loaded image as a numpy array
def load_image(file_path, gray=False, verbose=False):
    """
    Load an image from a file path into a numpy array.
    IN : file_path (str) - Path to the image file
         gray (bool) - If True, load the image in grayscale
         verbose (bool) - If True, print image information and show the image
    OUT : image (ndarray) - Loaded image as a numpy array
    """
    image = io.imread(file_path, as_gray=gray)
    if verbose:
        print(type(image))
        print("Image shape:", image.shape)
        print("Data type:", image.dtype)

        plt.imshow(image)
        plt.show()
    return image

def recadrage_dynamique(image, min_val, max_val):
    """
    Rescale the image pixels to the range [min_val, max_val].
    IN : image (ndarray) - Input image array
         min_val (float) - Minimum value for rescaling
         max_val (float) - Maximum value for rescaling
    OUT : rescaled_image (ndarray) - Rescaled image array
    """
    min_img = np.min(image)
    max_img = np.max(image)
    if max_img - min_img == 0:
        return image
    else:
        return (((max_val - min_val) / (max_img - min_img)) * (image - min_img)).astype(np.uint8)