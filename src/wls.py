# Algorithms using weighted least squares (WLS) framework to compute edge-preserving decomposition
# Author : Nino Rottier and Timothée Klein

# Imports
from scipy import sparse
from skimage import io, color, exposure
import numpy as np

# Algorithm of edge-preserving smoothing
# IN : input image, lambda, eps, nb_layers, alpha
# OUT : array of nb_layers images, each image is a smoothed version of the input image


def gradient(n,m):
    """ Gradient of the image
    IN : n - number of rows in the image
    m - number of column in the image
    OUT : Dx, Dy - sparse matrices of the gradient in x and y directions
    """
    nb_pixels = n * m
    eye = [1] * nb_pixels
    n_eye = [-1] * nb_pixels
    Dx = sparse.diags_array((n_eye, eye), (0, n), (nb_pixels, nb_pixels), format='csr')
    Dx[-n:,:] = 0
    Dy = sparse.diags_array((n_eye, eye), (0, 1), (nb_pixels, nb_pixels), format='csr')
    Dy[n-1:n:,:] = 0

    return Dx, Dy
    

def smoothness_matrixes(g, eps, alpha):
    """ Smoothness matrixes for the WLS algorithm
    IN : g - image, eps - epsilon parameter, alpha - alpha parameter
    OUT : Ax, Ay - sparse matrices of the smoothness in x and y directions
    """
    # Compute log-luminance channel of the input image g
    lab_image = color.rgb2lab(g)
    log_luminance = exposure.adjust_log(lab_image[:, :, 0])
    
    return log_luminance

