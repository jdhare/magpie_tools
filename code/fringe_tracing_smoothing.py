# The import corner
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_niblack, rank
from skimage.measure import find_contours
from skimage.morphology import thin
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
import warnings
from ipywidgets import FloatProgress
from IPython.display import display


# This loads an interferogram
def load_file(name):
    return rgb2gray(imread(name)), os.path.splitext(name)[0]


def blur(interferogram, blur):
    return gaussian(interferogram, blur)


def threshold(interferogram, window_size=51, k=0.3):
    thresh_niblack = threshold_niblack(interferogram, window_size=window_size, k=0.3)
    return interferogram > thresh_niblack


def contours(image):
    return find_contours(image, 0.5)


def smooth_contours(contours, range_len=10):
    f = FloatProgress(min=0, max=len(contours)-1)
    display(f)

    smoothed_contours = []
    for n, contour in enumerate(contours):
        smoothed_contour = []
        length = len(contour)
        if 500<length:
            for i in range(length):
                proposed = np.mean(np.take(contour, range(i-range_len, i+range_len), mode='wrap', axis=0), axis=0).astype('int')
                if len(smoothed_contour) != 0 and (proposed[0] != smoothed_contour[-1][0] or proposed[1] != smoothed_contour[-1][1]):
                    if (proposed[0]-smoothed_contour[-1][0])**2+(proposed[1]-smoothed_contour[-1][1])**2 > 2:
                        smoothed_contour.append([int(np.mean((proposed[0], smoothed_contour[-1][0]))), int(np.mean((proposed[1],smoothed_contour[-1][1])))])
                    smoothed_contour.append(proposed)
                elif len(smoothed_contour) == 0:
                    smoothed_contour.append(proposed)
            smoothed_contour = np.array(smoothed_contour)
            smoothed_contours.append(smoothed_contour)
            f.value = n
    f.value = len(contours)-1
    return np.array(smoothed_contours)


def paint_in(contours, interferogram):
    painted_in = np.zeros_like(interferogram)
    f = FloatProgress(min=0, max=len(contours)-1)
    display(f)
    limits = [[] for i in interferogram[0]]
    for n, contour in enumerate(contours):
        for i in range(len(contour)):
            colour = -1
            direction = contour[(i+1) % len(contour), 1]-contour[i-1, 1]
            if direction > 0:
                colour = 0
            elif direction < 0:
                colour = 1
            else:
                direction = ((contour[i, 1]-contour[i-1, 1])*(contour[i, 0]+contour[i-1, 0])
                             + (contour[(i+1) % len(contour), 1]-contour[i, 1])*(contour[(i+1) % len(contour), 0]+contour[i, 0]))
                if contour[(i+1) % len(contour), 0]-contour[i-1, 0]:
                    if direction > 0:
                        colour = 1
                    elif direction <= 0:
                        colour = 0
            if colour != -1:
                paint_limit = 0
                for limit in limits[contour[i, 1]]:
                    if limit < contour[i, 0] and paint_limit < limit:
                        paint_limit = limit
                painted_in[paint_limit+1:contour[i, 0], contour[i, 1]] = colour
                limits[contour[i, 1]].append(contour[i, 0])
                painted_in[contour[i, 0], contour[i, 1]] = 1
        f.value = n

    return painted_in


def thin_fringes(image, inverse=False):
    if inverse:
        print("Thinning dark fringes...")
        return thin(-1*(image-1), max_iter=100)
    else:
        print("Thinning bright fringes...")
        return thin(image, max_iter=100)


def save_file(fn, image):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(fn+'.png', (image*255).astype(int))


def save_file_alpha(fn, image, invert=False, colors=[255, 255, 255]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if invert:
            inv_image = -1*(image-1)
        else:
            inv_image = image
        imsave(fn+'.png', np.dstack((inv_image*colors[0],
                                     inv_image*colors[1],
                                     inv_image*colors[2],
                                     image*255)))


def highlight_errors(image):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rank.sum(image/255, np.ones((5, 5))) > 5


def plot_compare(fig, image0, image1, clim0=None, clim1=None):
    ax = ImageGrid(fig, rect=(0.08, 0.1, 0.8, 0.8),
                   nrows_ncols=(1, 2),
                   axes_pad=0.1,
                   share_all=True,
                   )
    if clim0 is not None:
    imshow0 = ax[0].imshow(image0, cmap='gray', clim=clim0)
    imshow1 = ax[1].imshow(image1, cmap='gray', clim=clim1)
        imshow1 = ax[1].imshow(image1, cmap='gray')
    return imshow0, imshow1
