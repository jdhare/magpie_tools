import matplotlib.pyplot as plt
import scipy as sp
from scipy.ndimage.interpolation import shift
import ipywidgets as widgets

def plot_shift(im0, im, shift_y,shift_x, lim):
    s=shift(im, [shift_y,shift_x])
    diff=s-im0
    fig, ax=plt.subplots(figsize=(16,16))
    ax.imshow(diff, cmap=plt.cm.bwr, clim=[-lim,lim])

sy=widgets.FloatSlider(min=-50,max=50,step=0.5,value=shifts[image-1,0])
sx=widgets.FloatSlider(min=-50,max=50,step=0.5,value=shifts[image-1,1])
li=widgets.FloatSlider(min=0,max=2000,step=10,value=1000)

