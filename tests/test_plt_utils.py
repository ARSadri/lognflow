#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest

import matplotlib.pyplot as plt
import lognflow
from lognflow.plt_utils import (
    plt_imshow, complex2hsv_colorbar, plt_imhist,complex2hsv)
import numpy as np

def test_numbers_as_images():
    dataset_shape = (10, 10, 64, 64)
    fontsize = 10
    dataset = lognflow.plt_utils.numbers_as_images_4D(
        dataset_shape, fontsize)

    ##########################################################################
    n_x, n_y, n_r, n_c = dataset_shape
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    /np.log(np.maximum(n_x, n_y))) + 1
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    for ind_x, ind_y in zip([0,     n_x//3, n_x//2, n_x-1], 
                            [n_x-1, n_x//2, n_x//3, 0    ]):
        plt.figure()
        plt.imshow(dataset[ind_x, ind_y], cmap = 'gray') 
        plt.title(number_text_base.format(ind_x = ind_x, ind_y = ind_y,
                                          width = txt_width))
    plt.show()

def test_plot_gaussian_gradient():
    print('test_plot_gaussian_gradient')
    x = np.arange(0, 2, 0.1)
    mu = x**2
    std = mu**0.5

    pgg = lognflow.plot_gaussian_gradient()
    pgg.addPlot(x = x, mu = mu, std = std, 
                  gradient_color = (1, 0, 0), 
                  label = 'red',
                  mu_color = (0.75, 0, 0, 1),
                  mu_linewidth = 3)
    pgg.show()

def test_pltfig_to_numpy():
    fig, ax = plt.subplots(111)
    ax[0].imshow(np.random.rand(100, 100))
    np_data = lognflow.plt_utils.pltfig_to_numpy(fig)
    print(np_data.shape)
    plt.close()

def test_imshow_series():
    data = [np.random.rand(10, 100, 100),
            np.random.rand(10, 10, 10)]
    lognflow.plt_utils.imshow_series(data)
    plt.show()

def test_imshow_by_subplots():
    data = np.random.rand(16, 100, 100)
    lognflow.plt_utils.imshow_by_subplots(data)
    plt.show()

def test_plt_imshow():
    data = np.random.rand(100, 100) + 1j * np.random.rand(100, 100)
    plt_imshow(data, cmap = 'complex')
    plt.show()
    
def test_complex2hsv_colorbar():
    complex2hsv_colorbar()
    plt.show()

def test_plt_imhist():
    img = np.zeros((100, 100))
    indsi, indsj = np.where(img == 0)
    mask = ((indsi - 30)**2 + (indsj - 30)**2)**0.5 > 15
    mask = mask.reshape(*img.shape)
    img[mask == 0] = np.random.randn(int((mask == 0).sum()))
    img[mask == 1] = 10 + np.random.randn(int((mask == 1).sum()))
    plt_imhist(img, 
               kwargs_for_imshow = {'cmap' : 'jet'}, 
               kwargs_for_hist = {'bins': 40})
    plt.show()

def test_plt_imshow_complex():
    
    # Define the meshgrid for testing
    comx, comy = np.meshgrid(np.arange(-7, 8, 1), np.arange(-7, 8, 1))
    com = comx + 1j * comy
    print(comx)
    print(comy)
    # Use the existing complex2hsv function to convert complex data to RGB
    img, data_abs, data_angle = complex2hsv(com)
    
    # Calculate min and max angles
    vmin = data_abs.min()
    vmax = data_abs.max()
    try:
        min_angle = data_angle[data_abs > 0].min()
    except:
        min_angle = 0
    try:
        max_angle = data_angle[data_abs > 0].max()
    except:
        max_angle = 0
    
    # Plot the complex image
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img, extent=(-7, 8, -7, 8))
    
    # Annotate each pixel with its corresponding comx and comy values
    for i in range(0, comx.shape[0], 1):
        for j in range(0, comx.shape[1], 1):
            ax.text(j - 7+0.5, -i + 7+0.5, f'({comx[i, j]}, {comy[i, j]})', ha='center', va='center', fontsize=8, color='white')
    
    # Create and plot the color disc as an inset
    fig, ax_inset = complex2hsv_colorbar((fig, ax.inset_axes([0.78, 0.08, 0.18, 0.18], transform=ax.transAxes)),
                                         vmin=vmin, vmax=vmax, min_angle=min_angle, max_angle=max_angle)
    ax_inset.patch.set_alpha(0)  # Make the background of the inset axis transparent
    
    plt.show()

if __name__ == '__main__':
    test_plt_imhist()
    test_plt_imshow_complex()
    test_complex2hsv_colorbar()
    test_imshow_by_subplots()
    test_plot_gaussian_gradient()
    test_plt_imshow()
    test_imshow_series()
    test_numbers_as_images()
    test_pltfig_to_numpy()
