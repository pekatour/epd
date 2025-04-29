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

    # Create Dx
    ex = np.ones(nb_pixels)
    ex[-n:] = 0
    Dx = sparse.diags_array([-ex, ex], offsets=[0, n], shape=(nb_pixels, nb_pixels), format='csr')

    # Create Dy
    ey = np.ones(nb_pixels)
    ey[n-1::n] = 0
    Dy = sparse.diags_array([-ey, ey], offsets=[0, 1], shape=(nb_pixels, nb_pixels), format='csr')

    return Dx, Dy

def smoothness_matrixes(g, Dx, Dy, alpha, eps):
    """ Smoothness matrixes for the WLS algorithm
    IN : g - image, eps - epsilon parameter, alpha - alpha parameter
    OUT : Ax, Ay - sparse matrices of the smoothness in x and y directions
    """
    # Compute log-luminance channel of the input image g
    lab_image = color.rgb2lab(g)
    log_luminance = exposure.adjust_log(lab_image[:, :, 0])
    
    # Compute matrix of derivatives of log-luminance
    Lx = Dx.dot(log_luminance.flatten())
    Ly = Dy.dot(log_luminance.flatten())
    Lx = np.abs(Lx)
    Ly = np.abs(Ly)

    # Compute Ax and Ay matrices
    Ax = np.power(Lx, alpha) + eps
    Ay = np.power(Ly, alpha) + eps
    Ax = 1 / Ax
    Ay = 1 / Ay
    n, m = g.shape[:2]
    Ax = sparse.diags_array(Ax, offsets=0, shape=(n * m, n * m), format='csr')
    Ay = sparse.diags_array(Ay, offsets=0, shape=(n * m, n * m), format='csr')

    return Ax, Ay

def lagrangian(Ax, Ay, Dx, Dy):
    """ Lagrangian of the WLS algorithm
    IN : Ax, Ay, Dx, Dy
    OUT : Lg - lagrangian matrix
    """
    return Dx.T.dot(Ax).dot(Dx) + Dy.T.dot(Ay).dot(Dy)

def iteration(g, lambda_, alpha, eps=0.0001):
    """ Iteration of the WLS algorithm
    IN : g - image, lambda - lambda parameter, alpha - alpha parameter, eps - epsilon parameter
    OUT : smoothed image
    """
    n, m = g.shape[:2]
    Dx, Dy = gradient(n, m)
    Ax, Ay = smoothness_matrixes(g, Dx, Dy, alpha, eps)
    Lg = lagrangian(Ax, Ay, Dx, Dy)
    Lg = Lg.tocsr()
    
    I = sparse.eye(n * m, format='csr')
    A = I + lambda_ * Lg

    return sparse.linalg.spsolve(A, g[:,:,0].flatten())
