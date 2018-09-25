import os
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.morphology import thin, skeletonize
from skimage.draw import circle
from skimage.filters import gaussian
from scipy.misc import imsave

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display

def create_filter(fft, R0, theta, radius_of_filter, blur):
    y_size, x_size=fft.shape
    X=round(R0*np.sin(theta*np.pi/180))
    Y=round(R0*np.cos(theta*np.pi/180))
    x1=round(x_size/2)-X
    x2=round(x_size/2)+X
    y1=round(y_size/2)+Y
    y2=round(y_size/2)-Y

    fft_filter=np.zeros_like(np.abs(fft))

    # two circle filter
    rr, cc = circle(r=y1, c=x1, radius=radius_of_filter, shape=fft_filter.shape)
    fft_filter[rr, cc] = 1
    rr, cc = circle(r=y2, c=x2, radius=radius_of_filter, shape=fft_filter.shape)
    fft_filter[rr, cc] = 1
    
    fft_filter=gaussian(fft_filter, blur)

    return (x1,x2,y1,y2), fft_filter

def plot_filter(fft, R0, theta, radius_of_filter, blur):
    (x1,x2,y1,y2), fft_filter=create_filter(fft, R0, theta, radius_of_filter, blur)
    
    x_l=int(min(x1,x2)-radius_of_filter)
    x_u=int(max(x1,x2)+radius_of_filter)
    y_l=int(min(y2,y1)-radius_of_filter)
    y_u=int(max(y1,y2)+radius_of_filter)
    
    masked_fft=fft_filter*fft
   
    fig, ax=plt.subplots(figsize=(8,8))
    ax.imshow(np.abs(masked_fft[y_l:y_u,x_l:x_u]))



def plot_threshold(im, threshold):
    bwimage=im>threshold
    imthin=thin(bwimage)

    fig, ax=plt.subplots(figsize=(12,8))
    ax.imshow(imthin, cmap='gray')
    
x_fringe_width=widgets.IntSlider(min=-200,
                                 max=200,
                                 step=5,
                                 value=100,
                                 description='Fringe width in x:', 
                                 continuous_update=False)

y_fringe_width=widgets.IntSlider(min=-200,
                                 max=200,
                                 step=5,
                                 value=100,
                                 description='Fringe width in y:', 
                                 continuous_update=False)

R0=widgets.IntSlider(min=0,
                                 max=200,
                                 step=5,
                                 value=100,
                                 description='R_0:', 
                                 continuous_update=False)

theta=widgets.IntSlider(min=0,
                                 max=180,
                                 step=5,
                                 value=100,
                                 description='Theta:', 
                                 continuous_update=False)


radius_of_filter=widgets.IntSlider(min=0,
                                   max=200,
                                   step=5,
                                   value=40,
                                   description='Radius of filter:', 
                                   continuous_update=False)

blur_edges=widgets.IntSlider(min=0,
                       max=50,
                       step=1,
                       value=10,
                       description='Filter blur:', 
                       continuous_update=False)

threshold=widgets.FloatSlider(min=0,
                              max=0.1,
                              step=0.0001,
                              value=0.03,
                              description='Binary threshold:',
                              continuous_update=False,
                              readout=True,
                              readout_format='.3f',
                             )