#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:28:48 2017

@author: avanetten
Plot goespatial positional uncertainty

Useful resources:
https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image
https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
"""

#######################
indir = '/path/to/geospatial_uncertainty'
#######################

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

###############################################################################
def gauss_2d(x, y, x0=0, y0=0, sigma_x=5, sigma_y=-1, minz=10**(-6)):
    '''
    Two dimensional Guassian, normalized to have area of 1.0.
    Values below minz will be set to 0.0
    https://en.wikipedia.org/wiki/Gaussian_function
    '''
    # set sigma_y = sigma_x if it is not explicitly set
    if sigma_y < 0:
        sigma_y = sigma_x
        
    # set normalization
    A = 1. / (2. * np.pi * sigma_x * sigma_y)
    
    # set function
    f = A * np.exp( -1. * ( ((x-x0)**2 / (2 * sigma_x**2)) \
                       + ((y-y0)**2 / (2 * sigma_y**2)) ) )
    
    # clip minimum value
    f[f <= minz] = 0
    
    return f

    
###############################################################################
def get_meshgrid(h, w):
    '''Define meshgrid with size h * w. 
    Delta^2 is the area of each grid cell'''

    x = np.linspace(0, w, w)
    y = np.linspace(0, h ,h)
    delta = x[1] - x[0]              
    xs, ys = np.meshgrid(x, y) 
    return xs, ys, delta


###############################################################################
def combine_probs(z_tot, z_update):
    '''Update probability map
    Compute probability that either z_tot or z_update has occurred'''
        
    prob_no_occur = (1. - z_tot) * (1. - z_update)
    return 1. - prob_no_occur
    

###############################################################################
def create_probability_map(points, h=500, w=500, max_alpha=0.5, verbose=False):
    '''Make probabilty map
    points are Gaussian blur values of format:
        point = [x0, y0, sigma_x, sigma_y, minz]
    h = map height
    w = map width
    max_alpha is the desired value of the max alpha, used to rescale the
        output map (set < 0 to skip)'''

    # get grid
    xs, ys, delta = get_meshgrid(h, w)
    
    z_tot = np.zeros((h, w))
    # update probability map
    for i,p in enumerate(points):
        [x0, y0, sigma_x, sigma_y, minz] = p
        z_update = gauss_2d(xs, ys, x0=x0, y0=y0, sigma_x=sigma_x, 
                            sigma_y=sigma_y, minz=minz)
        # combine update and total
        z_tot = combine_probs(z_tot, z_update)
        
        if verbose:
            print i, "point: x0, y0:", x0, y0
            print "  sigma_x, sigma_y:", sigma_x, sigma_y
            print "  np.min(z_update):", np.min(z_update)
            print "  np.max(z_update):", np.max(z_update)
            # ensure Gaussian is corretly normalized (volume should be 1.0)
            print "  Volume of 2-D Gaussian:", np.sum(z_update) / delta**2
                      
    if verbose:
        print "Final map:"
        print "  np.min(z_tot):", np.min(z_tot)
        print "  np.max(z_tot):", np.max(z_tot)
        print "  Volume of 2-D Gaussian:", np.sum(z_tot) / delta**2
            
    ############ 
    # rescale z such that max alpha is the desired number
    if max_alpha > 0:
        scale = max_alpha / np.max(z_tot)
        if verbose:
            print "Rescale map to have max prob of:", max_alpha
            print "  Scaling factor:", scale
        z_tot *= scale         
             
    return z_tot

###############################################################################
def standard_uncertainty_plot(points, ax2, im=[], alpha_max=0.75,
                              facecolor='blue', 
                              title='Standard Uncertainty Vis', 
                              verbose=False):
    '''Plot standard method for uncertainty
    Sst ellipse size proportional to sigma, and opacity inversely proportional
    to sigma
    good reference:
        https://matplotlib.org/examples/pylab_examples/ellipse_demo.html'''
        
    if len(im) > 0:
        ax2.imshow(im)
        
    # iterate through points, and gather ellipses and sizes
    ells = []
    sizes = []
    for p in points:
        [x0, y0, sigma_x, sigma_y, minz] = p
        el_tmp = Ellipse(xy=(x0, y0), width=sigma_x, height=sigma_y, angle=0) 
        ells.append(el_tmp)
        size = np.pi * sigma_x * sigma_y
        sizes.append(size)
    
    # determine alpha from sizes
    alpha_scale = alpha_max * np.min(sizes) 
    alphas = []
    for s in sizes:
        a = alpha_scale / s
        alphas.append(a)
    if verbose:
        print "sizes:", sizes
        print "alphas:", alphas
    
    # Finally, make the plot
    for el_tmp, alpha in zip(ells, alphas):

        ax2.add_artist(el_tmp)
        el_tmp.set_alpha(alpha)
        el_tmp.set_facecolor(facecolor)
        
    ax2.axis('off')
    if len(title) > 0:
        ax2.set_title(title)
    
    return
    
###############################################################################
def plot_uncertainty(z_tot, points, im=[], outfile='', figsize=(15,4),
                   max_alpha=0.75, dpi=300):
    '''Plot positional uncertainty
    ax0: set opacity as equal to probability
    ax1: grayscale map with color between 0 (white) and 1 (black)'''

    plt.close('all')
    
    ############
    # set color channel to red
    r = np.ones(z_tot.shape)
    b, g = np.zeros(z_tot.shape), np.zeros(z_tot.shape)
    img_RGBA = cv2.merge((r, g, b, z_tot))
    #cv2.imwrite("test.png", img_RGBA)
        
    ############
    # make plots
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, 
         figsize=figsize)
 

    ############
    # plot standard method for uncertainty
    standard_uncertainty_plot(points, ax0, im=im, alpha_max=max_alpha,
                              facecolor='red', 
                              title='Standard Uncertainty Vis')
    plt.tight_layout()
                   
    ############
    # alpha channel (underlaying doesn't matter since no prob map overwrites)
    cmap = 'YlOrRd'  #'hot'  # 'binary', 'Wistia', 'Blues'
    ax1.imshow(z_tot, cmap=cmap)#, vmin=0, vmax=1)
    ax1.axis('off')
    ax1.set_title('Probability Map')
    plt.tight_layout()
    
    ############
    # img_rgba
    # Underlay background image, if desired
    if len(im) > 0:
        ax2.imshow(im)
    ax2.imshow(img_RGBA, cmap='Blues')
    ax2.axis('off')
    #ax0.set_title('RGBA Image')
    ax2.set_title('Gaussian Uncertainty Vis')    
    
    ############
    #plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if len(outfile) > 0:
        plt.savefig(outfile, dpi=dpi)

    return

###############################################################################
def plot_uncertainty_indiv(z_tot, points, im=[], outdir='', figsize=(4,4),
                   max_alpha=0.75, dpi=300):
    '''Plot positional uncertainty
    ax0: set opacity as equal to probability
    ax1: grayscale map with color between 0 (white) and 1 (black)'''

    plt.close('all')
    
    ############
    # set color channel to red
    r = np.ones(z_tot.shape)
    b, g = np.zeros(z_tot.shape), np.zeros(z_tot.shape)
    img_RGBA = cv2.merge((r, g, b, z_tot))
    #cv2.imwrite("test.png", img_RGBA)
        
    ############
    # plot standard method for uncertainty
    # make plots
    fig, ax0 = plt.subplots(1, 1)
    standard_uncertainty_plot(points, ax0, im=im, alpha_max=max_alpha,
                              facecolor='red', title='')
    ax0.axis('off')
    plt.tight_layout()
    outname = os.path.join(outdir, 'ellipse_uncertainty.png')
    plt.savefig(outname, dpi=dpi, bbox_inches='tight', pad_inches=0)

                  
    ############
    # alpha channel (underlaying doesn't matter since no prob map overwrites)
    fig, ax1 = plt.subplots(1, 1)
    cmap = 'YlOrRd'  #'hot'  # 'binary', 'Wistia', 'Blues'
    ax1.imshow(z_tot, cmap=cmap)#, vmin=0, vmax=1)
    ax1.axis('off')
    plt.tight_layout()
    outname = os.path.join(outdir, 'gauss_probability_map.png')
    plt.savefig(outname, dpi=dpi, bbox_inches='tight', pad_inches=0)

    ############
    # img_rgba
    # Underlay background image, if desired
    fig, ax2 = plt.subplots(1, 1)
    if len(im) > 0:
        ax2.imshow(im)
    ax2.imshow(img_RGBA, cmap='Blues')
    ax2.axis('off')
    outname = os.path.join(outdir, 'gauss_uncertainty.png')
    plt.savefig(outname, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    return
###############################################################################
###############################################################################
def main():
    '''Execute test'''
    
    N_points = 16 #4
    verbose = True
    max_alpha = 0.66
    
    ############
    # define background image
    im_loc = os.path.join(indir, 'background_images/jpl_moscow.jpg')
    #im_loc = os.path.join(indir, 'background_images/Map_of_the_Battle_of_the_Somme.png')
    
    # define output image
    outfile = os.path.join(indir, 'outplot.png')
                          
    # get image
    im = cv2.imread(im_loc, 1)
    #im = cv2.resize(im, xs.shape)   # reshape if desired
    h, w, nbands = im.shape
    
    ############
    # get meshgrid
    xs, ys, delta = get_meshgrid(h, w)
    
    ############
    # define points         
    #   point = [x0, y0, sigma_x, sigma_y, minz]
    points = []
    minz = 10**(-9)
    xmin, xmax = int(0.15*w), int(0.85*w)
    ymin, ymax = int(0.15*h), int(0.85*h)
    sigma_min, sigma_max = int(0.015*w), int(0.03*w)
    #sigma_min, sigma_max = int(0.1*w), int(0.2*w)

    for i in range(N_points):
        x0 = np.random.randint(xmin, xmax)
        y0 = np.random.randint(ymin, ymax)
        sigma_x = np.random.randint(sigma_min, sigma_max)
        sigma_y = np.random.randint(sigma_min, sigma_max)
        points.append([x0, y0, sigma_x, sigma_y, minz])
    
    ############
    # get probability map
    probability_map = create_probability_map(points, h=h, w=w, 
                                             max_alpha=max_alpha, 
                                              verbose=verbose)
    ############
    plot_uncertainty(probability_map, points, im=im, outfile=outfile, 
                     figsize=(12,4))

    #plot_uncertainty_indiv(probability_map, points, im=im, outdir=indir, 
    #                 figsize=(4,4))
###############################################################################
if __name__ == "__main__":
    main()