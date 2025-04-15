# Algorithms using weighted least squares (WLS) framework to compute edge-preserving decomposition
# Author : Nino Rottier and Timothée Klein

# Imports
from scipy import sparse
from skimage import color, exposure
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
    # Create the diagonals
    e = np.ones(nb_pixels)

    # Dx: horizontal differences
    data_dx = np.vstack((-e, e))
    offsets_dx = np.array([0, n])
    Dx = sparse.dia_matrix((data_dx, offsets_dx), shape=(nb_pixels, nb_pixels)).tolil()
    Dx[-n:, :] = 0  # Zero out the last n rows
    Dx = Dx.tocsr()  # Convert to CSR for efficient use

    # Dy: vertical differences
    data_dy = np.vstack((-e, e))
    offsets_dy = np.array([0, 1])
    Dy = sparse.dia_matrix((data_dy, offsets_dy), shape=(nb_pixels, nb_pixels)).tolil()
    Dy[n-1::n, :] = 0  # Zero out every n-th row
    Dy = Dy.tocsr()  # Convert to CSR for efficient use
    return Dx, Dy

def smoothness_matrixes(g, alpha, eps=0.0001):
    """ Smoothness matrixes for the WLS algorithm
    IN : g - image, eps - epsilon parameter, alpha - alpha parameter
    OUT : Ax, Ay - sparse matrices of the smoothness in x and y directions
    """
    # Compute log-luminance channel of the input image g
    lab_image = color.rgb2lab(g)
    log_luminance = exposure.adjust_log(lab_image[:, :, 0])
    
    # Compute matrix of derivatives of log-luminance
    n, m = log_luminance.shape
    Dx, Dy = gradient(n, m)
    Lx = Dx.dot(log_luminance.flatten())
    Ly = Dy.dot(log_luminance.flatten())
    Lx = np.abs(Lx)
    Ly = np.abs(Ly)

    # Compute Ax and Ay matrices
    Ax = np.power(Lx, alpha) + eps
    Ay = np.power(Ly, alpha) + eps
    Ax = 1 / Ax
    Ay = 1 / Ay

    return Ax, Ay




    

