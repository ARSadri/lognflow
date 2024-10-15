#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest
import time
import matplotlib.pyplot as plt
import lognflow
from lognflow.plt_utils import (
    plt_imshow, complex2hsv_colorbar, plt_imhist,complex2hsv,
    transform3D_viewer, plt_mark, plt_contours, question_dialog,
    plt_plot)
import numpy as np

def test_transform3D_viewer():
    in_pointcloud = np.random.randn(100, 3)
    moving_inds = np.where((in_pointcloud[:, 0] > 0) & 
                           (in_pointcloud[:, 1] > 0) & 
                           (in_pointcloud[:, 2] > 0))[0]
    points_classes = np.ones(len(in_pointcloud))
    points_classes[moving_inds] = 0
    in_pointcloud2 = in_pointcloud[moving_inds].copy()
    tp = transform3D_viewer(in_pointcloud, points_classes)
    plt.show()
        
    in_pointcloud2_transformed = tp.apply(in_pointcloud2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(tp.PC[moving_inds, 0], 
               tp.PC[moving_inds, 1],
               tp.PC[moving_inds, 2], color = 'green')
    
    ax.scatter(in_pointcloud2[:, 0], 
               in_pointcloud2[:, 1], 
               in_pointcloud2[:, 2], color = 'blue')
    
    ax.scatter(in_pointcloud2_transformed[:, 0]+0.05, 
               in_pointcloud2_transformed[:, 1]+0.05, 
               in_pointcloud2_transformed[:, 2]+0.05,
               marker = 's', color = 'red')

    plt.show()    
    
    tp.figure()
    plt.show()
    print(tp.PC[moving_inds].mean(0))

    
def test_numbers_as_images():
    dataset_shape = (10, 10, 64, 64)
    fontsize = 10
    dataset = lognflow.plt_utils.numbers_as_images_4D(
        dataset_shape, fontsize)

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

    pgg = lognflow.plt_utils.plot_gaussian_gradient()
    pgg.addPlot(x = x, mu = mu, std = std, 
                  gradient_color = (1, 0, 0), 
                  label = 'red',
                  mu_color = (0.75, 0, 0, 1),
                  mu_linewidth = 3)
    
    pgg = lognflow.plt_utils.plot_gaussian_gradient()
    pgg.addPlot(x = x, mu = mu, std = std, 
                  gradient_color = (1, 0, 0), 
                  label = 'red',
                  mu_color = (0.75, 0, 0, 1),
                  mu_linewidth = 1)
    pgg.addPlot(x = x, mu = 8 - mu, std = std, 
                gradient_color = (0, 0, 1), 
                label = 'red',
                mu_color = (0, 0, 0.75, 1),
                mu_linewidth = 1)    
    pgg.show()

def test_plt_fig_to_numpy():
    fig, ax = plt.subplots(111)
    ax[0].imshow(np.random.rand(100, 100))
    np_data = lognflow.plt_utils.plt_fig_to_numpy(fig)
    print(np_data.shape)
    plt.close()

def test_plt_imshow_series():
    data = [np.random.rand(10, 100, 100),
            np.random.rand(10, 10, 10)]
    lognflow.plt_utils.plt_imshow_series(data)
    plt.show()

def test_plt_imshow_subplots():
    data = np.random.rand(15, 100, 100, 3)
    lognflow.plt_utils.plt_imshow_subplots(data, colorbar = False)

    data = [np.random.rand(100, 100), 
            np.random.rand(100, 150), 
            np.random.rand(50, 100)]
    lognflow.plt_utils.plt_imshow_subplots(data)

    data = np.random.rand(15, 100, 100)
    grid_locations = (np.random.rand(len(data), 2)*1000).astype('int')
    lognflow.plt_utils.plt_imshow_subplots(
        data, grid_locations = grid_locations)
    
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
    
    comx, comy = np.meshgrid(np.arange(-7, 8, 1), np.arange(-7, 8, 1))
    com = comx + 1j * comy
    print(comx)
    print(comy)
    img, data_abs, data_angle = complex2hsv(com)
    
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
    
    fig, ax = plt_imshow(img, extent=(-7, 8, -7, 8), title = 'complex2hsv',
                         colorbar = False)
    
    for i in range(0, comx.shape[0], 1):
        for j in range(0, comx.shape[1], 1):
            ax.text(j - 7+0.5, -i + 7+0.5, f'({comx[i, j]}, {comy[i, j]})', 
                    ha='center', va='center', fontsize=8, color='white')
    
    fig, ax_inset = complex2hsv_colorbar(
        (fig, ax.inset_axes([0.79, 0.03, 0.18, 0.18], transform=ax.transAxes)),
            vmin=vmin, vmax=vmax, min_angle=min_angle, max_angle=max_angle)
    ax_inset.patch.set_alpha(0)
    
    plt_imshow(np.random.rand(100, 100) + 1j * np.random.rand(100, 100),
               cmap = 'gray_real_imag')
    
    plt_imshow(np.random.rand(100, 100) + 1j * np.random.rand(100, 100),
               cmap = 'jet_real_imag')

    plt_imshow(np.random.rand(100, 100) + 1j * np.random.rand(100, 100))
    
    plt.show()

def test_plt_mark():
    coords = np.random.rand(1000, 2)*100
    fig, ax, markersize = plt_mark(
        coords, fig_ax=None, figsize=None,
        markersize=None, return_markersize = True)
    for cnt in range(23):
        coords = np.array([np.arange(1 + 1*cnt,1+ 1*(cnt+1)), np.zeros(1)]).T
        fig_ax = plt_mark(coords, fig_ax=(fig, ax), markersize=markersize)
    
    plt.show()

def test_plt_contours():
    yy, xx = np.meshgrid(np.arange(100), np.arange(50))
    Z_list = [np.exp(- ((xx-15)**2 + (yy-30)**2)**0.5 / 18), 
              np.exp(- ((xx-35)**2 + (yy-70)**2)**0.5 / 18) ]
    plt_contours(Z_list)
    plt.show()

def test_question_dialog():
    vec = np.random.rand(100)
    img = np.random.rand(100, 100)
    question = 'how good is it?'
    
    result = question_dialog(vec)
    print(result)
    result = question_dialog(img)
    print(result)
    result = question_dialog(question)
    print(result)

def test_stack_to_frame():
   data4d = np.random.rand(25, 32, 32, 3)
   img = lognflow.plt_utils.stack_to_frame(data4d, borders = np.nan)
   plt.figure()
   plt.imshow(img)
   
   data4d = np.random.rand(32, 32, 16, 16, 3)
   stack = data4d.reshape(-1, *data4d.shape[2:])
   frame = lognflow.plt_utils.stack_to_frame(stack, borders = np.nan)
   plt.figure()
   im = plt.imshow(frame)
   lognflow.plt_utils.plt_colorbar(im)
   plt.show()

def test_plt_plot():
    y_values_list = [[1, 2, 3], [4, 5, 6]]
    x_values_list = [[1, 2, 3]]
    plt_plot(y_values_list, '-*', x_values_list = x_values_list)
    plt_plot(y_values_list, '-*')
    plt.show()

if __name__ == '__main__':
    test_plt_plot()
    test_plt_imshow_complex()
    test_question_dialog()
    test_plt_contours()
    test_complex2hsv_colorbar()
    test_plt_mark()
    test_plot_gaussian_gradient()
    test_transform3D_viewer()
    test_plt_imshow_subplots()
    test_plt_imhist()
    test_plt_imshow()
    test_plt_imshow_series()
    test_numbers_as_images()
    test_plt_fig_to_numpy()
    test_stack_to_frame()