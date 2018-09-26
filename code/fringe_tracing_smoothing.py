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


# This function loads an interferogram and returns its filename
def load_file(name):
    return rgb2gray(imread(name)), os.path.splitext(name)[0]


# This function blurs the given image
def blur(image, blur):
    return gaussian(image, blur)


# This function performs Niblack thresholding on the given image.
# http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html
def threshold(image, window_size=51, k=0.3):
    thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.3)
    return image > thresh_niblack


# This function returns the contours of a binary image (the 0.5 is inbetween
# black and white)
def contours(image):
    return find_contours(image, 0.5)


# This is the custom function that performs contour smoothing
def smooth_contours(contours, range_len=10, limit_len=500):
    # Create a progress bar
    f = FloatProgress(min=0, max=len(contours)-1)
    display(f)

    smoothed_contours = []
    # Iterate over all contours
    for n, contour in enumerate(contours):
        smoothed_contour = []
        length = len(contour)
        # Reject contours that are too short
        if limit_len < length:
            # Go over each point in the contour
            for i in range(length):
                # Calculate the new position of the point as the mean
                # of the positions of the neighbours.
                # np.take is needed for wrap-around indexing
                proposed = np.mean(np.take(contour,
                                           range(i-range_len, i+range_len),
                                           mode='wrap', axis=0),
                                           axis=0).astype('int')
                # We now check whether this point is not the same as the
                # last one (when averaging and rounding, points tend to overlap)
                if len(smoothed_contour) != 0 and (proposed[0] != smoothed_contour[-1][0] or proposed[1] != smoothed_contour[-1][1]):
                    # We also check the distance between the new point and
                    # the previous one. We want them to be neighbours.
                    if (proposed[0]-smoothed_contour[-1][0])**2+(proposed[1]-smoothed_contour[-1][1])**2 > 2:
                        # This is a naive fix, but works for most cases
                        smoothed_contour.append([int(np.mean((proposed[0], smoothed_contour[-1][0]))), int(np.mean((proposed[1],smoothed_contour[-1][1])))])
                    smoothed_contour.append(proposed)
                elif len(smoothed_contour) == 0:
                    smoothed_contour.append(proposed)
            smoothed_contour = np.array(smoothed_contour)
            smoothed_contours.append(smoothed_contour)
            # Update the progress bar
            f.value = n
    # The progress bar is updated only when processing long enough contours
    # (for performance reasons). Set it so that it's full, to indicate
    # completion of the process
    f.value = len(contours)-1
    return np.array(smoothed_contours)


# It's hard to believe, but I didn't find a standard function that did this.
# Correction: did this quickly. skimage's draw.polygon is painfully slow,
# especially for the long and often aggressively concave polygons.
# So here we are, with a custom function which makes the pixels inside
# of a contour white
def paint_in(contours, image):
    # Create an empty array the size of the original image
    painted_in = np.zeros_like(image)
    # Create the progress bar
    f = FloatProgress(min=0, max=len(contours)-1)
    display(f)
    # Make a list of lists of painted in points for every column of the image.
    # Those function as limits - we paint from a certain position up to the
    # closest one of these
    limits = [[] for i in image[0]]
    # Paint in every contour
    for n, contour in enumerate(contours):
        # Go through each point of the contour
        for i in range(len(contour)):
            # If colour is -1 by the end of the forthcoming ifs,
            # that means the direction in which the contour is going
            # is too ambiguous to use
            colour = -1
            # Determine if the contour is going left of right.
            # This uses a very convenient aspect of skimage's contour-finding
            # function - they're either clockwise or anticlockwise depending
            # on the colour they enclose.
            # Note that we usually compare the point before and the point
            # after, to get a general trend at that position.
            direction = contour[(i+1) % len(contour), 1]-contour[i-1, 1]
            if direction > 0:
                colour = 0
            elif direction < 0:
                colour = 1
            else:
                # If the x coordinate doesn't change, perform other checks:
                # This calculates the clockwise or anticlockwise direction
                direction = ((contour[i, 1]-contour[i-1, 1])*(contour[i, 0]+contour[i-1, 0])
                             + (contour[(i+1) % len(contour), 1]-contour[i, 1])*(contour[(i+1) % len(contour), 0]+contour[i, 0]))
                # Check that the y coordinate changes
                if contour[(i+1) % len(contour), 0]-contour[i-1, 0]:
                    if direction > 0:
                        colour = 1
                    elif direction <= 0:
                        colour = 0
            # If we have established what colour we want, paint the pixels
            # above this one
            if colour != -1:
                # Establish the painting limit, which is the highest value in
                # paint_limit for this column that is below the current pixel
                paint_limit = 0
                for limit in limits[contour[i, 1]]:
                    if limit < contour[i, 0] and paint_limit < limit:
                        paint_limit = limit
                # Paint in
                painted_in[paint_limit+1:contour[i, 0], contour[i, 1]] = colour
                # Add this pixel to the limit list
                limits[contour[i, 1]].append(contour[i, 0])
                # Paint this pixel white, so that the contours are always white
                painted_in[contour[i, 0], contour[i, 1]] = 1
        f.value = n
    # Return the finished image
    return painted_in


# Thin the fringes image
def thin_fringes(image, inverse=False):
    if inverse:
        # If we want to trace the inverse, we invert the image with some
        # simple maths
        print("Thinning dark fringes...")
        return thin(-1*(image-1), max_iter=100)
    else:
        print("Thinning bright fringes...")
        return thin(image, max_iter=100)


# This function saves the image data under the name provided as fn
def save_file(fn, image):
    # Silence the warning about low contrast that usually happens,
    # as well as any warnings about loosing accuracy during float->integer
    # conversion (insignificant here)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(fn+'.png', (image*255).astype(int))


# This function saves the image data, but makse everything
# that's false trarnsparent.
# Setting the colors (damned american standards in coding) argument
# allows for colouring the data (red for example)
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


# This function highlights possible errors in the tracing. It counts white
# pixels in a 5x5 square surrounding each pixels, and if there's more than
# 5, it is likely that branching occured in that region.
# Example:
#    oXooX
#    ooXXo
#    ooXoo
#    oooXo
#    ooooX - 7 white pixels, clear branching
def highlight_errors(image):
    # There is a type conversion warning that can be ignored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rank.sum(image/255, np.ones((5, 5))) > 5


# This function draws a side by side comparison graph of two images.
# They can be zoomed and panned together
def plot_compare(fig, image0, image1, clim0=None, clim1=None):
    # Create the axes using the ImageGrid class
    ax = ImageGrid(fig, rect=(0.08, 0.1, 0.8, 0.8),
                   nrows_ncols=(1, 2),
                   axes_pad=0.1,
                   share_all=True,
                   )
    # imshow the images, taking into account the colour limits if provided
    imshow0 = ax[0].imshow(image0, cmap='gray', clim=clim0)
    imshow1 = ax[1].imshow(image1, cmap='gray', clim=clim1)
    return imshow0, imshow1
