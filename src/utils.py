from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_image(file_path, gray=False, verbose=False):
    """
    Load an image from a file path into a numpy array.
    IN : file_path (str) - Path to the image file
         gray (bool) - If True, load the image in grayscale
         verbose (bool) - If True, print image information and show the image
    OUT : image (ndarray) - Loaded image as a numpy array
    """
    image = io.imread(file_path, as_gray=gray)
    if gray:
        image = image*255
    if verbose:
        print(type(image))
        print("Image shape:", image.shape)
        print("Data type:", image.dtype)

        plt.imshow(image, cmap='gray' if gray else None)
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
    

def epi(image_ref, image_filtered):
    """
    Compute the edge-preserving index (EPI) between two gray images.
    IN : image_ref (ndarray) - Reference image
         image_filtered (ndarray) - Filtered image
    OUT : epi_value (float) - EPI value
    """
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_ref_x = cv2.filter2D(image_ref, -1, sobel_x)
    gradient_ref_y = cv2.filter2D(image_ref, -1, sobel_y)
    gradient_ref = np.sqrt(gradient_ref_x**2 + gradient_ref_y**2)

    gradient_filtered_x = cv2.filter2D(image_filtered, -1, sobel_x)
    gradient_filtered_y = cv2.filter2D(image_filtered, -1, sobel_y)
    gradient_filtered = np.sqrt(gradient_filtered_x**2 + gradient_filtered_y**2)

    diff_ref = gradient_ref - np.mean(gradient_ref)
    diff_filtered = gradient_filtered - np.mean(gradient_filtered)

    epi_value = np.sum(diff_ref * diff_filtered) / (np.sqrt(np.sum(diff_ref**2)) * np.sqrt(np.sum(diff_filtered**2)))
    return epi_value
