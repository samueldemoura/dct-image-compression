import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def dct(x, inverse=False):
    '''Applies DCT transformation to input array x, returns array of coeficients.'''
    N = len(x)
    X = [0]*N

    for k in range(0, N):
        if k == 0:
            ck = math.sqrt(0.5)
        else:
            ck = 1

        _sum = 0
        for n in range(0, N):
            theta = 2*math.pi * (k / (2*N)) * n + (k*math.pi) / (2*N)
            _sum += x[n] * math.cos(theta)
            if inverse:
                _sum *= ck
        
        X[k] = math.sqrt(2/N) * _sum
        if not inverse:
            X[k] *= ck

    return X

def dct_2d(x, w, h, inverse=False):
    '''Applies DCT transformation repeatedly for each row and column.'''
    X = np.array([0]*(w * h)).reshape(w, h)

    for j in range(0, h):
        X[:, j] = dct(x[:, j], inverse)

    for i in range(0, w):
        X[i, :] = dct(x[i, :], inverse)
    
    return X

def dct_image(im, channels, inverse=False):
    '''Performs 2D DCT on a PIL Image object.''' 
    w = im.size[0]
    h = im.size[1]
    im_new = np.array([0]*(w * h * channels)).reshape(w, h, channels)

    for c in range(0, channels):
        im_data = np.array(im.getdata(c)).reshape(w, h)
        channel_data = dct_2d(im_data, w, h, inverse)
        # im_new[:][:][c] = channel_data (doesn't work)

        # mother of god why
        for x in range(0, w):
            for y in range(0, h):
                #if channel_data[x, y] > 255:
                #    channel_data[x, y] = 255

                #channel_data[x, y] = np.round(channel_data[x, y]).astype(int)
                im_new[x, y, c] = channel_data[x, y]

    return Image.fromarray(im_new)

# Testing
im = Image.open('test.png')
im_freq = dct_image(im, 4)
im_back = dct_image(im_freq, 4, True)
im_back.show()