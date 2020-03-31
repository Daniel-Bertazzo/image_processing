'''
Daniel Penna Chaves Bertazzo - 10349561

SCC0251 - Image Processing - ICMC/USP
First semester of 2020

Assignment 1: Intensity transformations
'''

import numpy as np
import imageio


# ..:: Transformation functions ::..

# 1) Inversion
def invert(r):
    '''
    Inverts a given image.
    Input:
        r: original image to be inverted
    Return:
        image after transformation
    '''
    return (255 - r).astype(np.uint8)    


# 2) Contrast modulation
def contrast_modulation(r, c, d):
    '''
    Computes a contrast modulation transformation on the image, resulting in an image with new
    highest and lowest values (brightest and darkest pixels). 
    Input:
        r: original image
        c: new lowest pixel value
        d: new highest pixel value
    Return:
        image after transformation
    '''
    # Finds minimum and maximum values in image
    a = np.amin(r)
    b = np.amax(r)
    
    # Computes the transformation
    return ((r-a) * ((d-c) / (b-a)) + c).astype(np.uint8)


# 3) Logarithmic function
def logarithmic(r):
    '''
    Computes a logarithmic transformation on the image.
    Input:
        r: original image
    Return:
        image after transformation
    '''
    # Finds the image's maximum pixel value    
    R = float(np.amax(r))
    
    # Computes the transformation
    return ( 255.0 * (np.log2(1.0 + r) / np.log2(1.0 + R)) ).astype(np.uint8)


# 4) Gamma adjustment
def gamma_adjustment(r, W, Lambda):
    '''
    Computes a gamma adjustment transformation in the image
    Input:
        r: original image
        W: weighs the result
        Lamdbda: transformation parameter
    Return:
        image after transformation
    '''
    return (W * np.power(r, Lambda)).astype(np.uint8)


# Root squared error (RSE)
def rse(r, m):
    '''
    Computes the Root Squared Error (RSE) between the original image and the
    image generated after the transformation.
    Input:
        r: original image
        m: modified image
    Return:
       root squared error value 
    '''
    return np.sqrt(np.sum(np.power(m.astype(np.float64) - r.astype(np.float64), 2)))



# *********************************************** "Main" ***********************************************

# ..:: Reading input data ::..
filename = str(input()).rstrip() # Image's file name
r = imageio.imread(filename)     # Reads the image
T = int(input())                 # Which transformation to apply
S = int(input())                 # Parameter S (to save or not afterwards)


# ..:: Selects which transformation function to call ::..

# Tranformation 1: Inversion
if T == 1:
    m = invert(r)

# Transformation 2: Contrast modulation
elif T == 2:
    # Reads parameters needed for transformation 2
    c = int(input()) # New lowest
    d = int(input()) # New highest
    
    m = contrast_modulation(r, c, d)

# Transformation 3: Logarithmic function
elif T == 3:
    m = logarithmic(r)

# Transformation 4: Gamma adjustment
elif T == 4:
    # Reads parameters needed for transformation 4
    W = int(input()) 
    Lambda = float(input())
    
    m = gamma_adjustment(r, W, Lambda)


# Computes the RSE between the images
err = rse(r, m)

# If S == 1, saves the transformed image
if S == 1:
    imageio.imwrite('transformed_img.png', m)

# Prints the output (the error)
print("%.4f" %err, end='')