from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt

def plt_colorbar(mappable):
    """ Add colobar to the current axis 
        This is specially useful in plt.subplots
        stackoverflow.com/questions/23876588/
            matplotlib-colorbar-in-each-subplot
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar

def plt_hist(vectors_list, 
             n_bins = 10, alpha = 0.5, normalize = False, 
             labels_list = None, **kwargs):
    if not (type(vectors_list) is list):
        vectors_list = [vectors_list]
    for vec_cnt, vec in enumerate(vectors_list):
        bins, edges = np.histogram(vec, n_bins)
        if normalize:
            bins = bins / bins.max()
        plt.bar(edges[:-1], bins, 
                width =np.diff(edges).mean(), alpha=alpha)
        if labels_list is None:
            plt.plot(edges[:-1], bins, **kwargs)
        else:
            assert len(labels_list) == len(vectors_list)
            plt.plot(edges[:-1], bins, 
                     label = f'{labels_list[vec_cnt]}', **kwargs)
            
def pltfig_to_numpy(fig):
    """ from https://www.icare.univ-lille.fr/how-to-
                    convert-a-matplotlib-figure-to-a-numpy-array-or-a-pil-image/
    """
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.ubyte)
    buf.shape = (w, h, 4)
    buf = np.roll (buf, 3, axis = 2)
    return buf

def numbers_as_images_3D(dataset_shape: tuple, fontsize, verbose = False):
    """ Numbers3D
    This function generates a 4D dataset of images with shape
    (n_x, n_r, n_c) where in each image the value "x" is written as a text
    that fills the image. As such, later when working with such a dataset you can
    look at the image and know which index it had before you use it.
    
    Follow this recipe to make good images:
    
    1- set n_x to 10, Set the desired n_r, n_c and width. 
    2- find fontsize that is the largest and still fits
    3- Increase n_x to desired size.
    
    You can provide a logs_root, log_dir or simply select a directory to save the
    output 3D array.
    
    """
    n_x, n_r, n_c = dataset_shape
    dataset = np.zeros(dataset_shape)    
    txt_width = int(np.log(n_x)/np.log(n_x)) + 1
    number_text_base = '{ind_x:0{width}}}'
    if(verbose):
        pBar = printprogress(n_x)
    for ind_x in range(n_x):
        mat = np.ones((n_r, n_c))
        number_text = number_text_base.format(ind_x = ind_x, 
                                              width = txt_width)
        fig = plt.figure(figsize = (1, 1))
        ax = fig.add_subplot(111)
        ax.imshow(mat, cmap = 'gray', vmin = 0, vmax = 1)
        ax.text(mat.shape[0]//2 - fontsize, mat.shape[1]//2 ,
                number_text, fontsize = fontsize)
        ax.axis('off')
        buf = pltfig_to_numpy(fig)
        plt.close()
        buf2 = buf[::buf.shape[0]//n_r, ::buf.shape[1]//n_c, :3].mean(2)
        buf2 = buf2[:n_r, :n_c]
        dataset[ind_x] = buf2.copy()
        if(verbose):
            pBar()
    return dataset

def numbers_as_images_4D(dataset_shape: tuple, fontsize, verbose = False):
    """ Numbers4D
    This function generates a 4D dataset of images with shape
    (n_x, n_y, n_r, n_c) where in each image the value "x, y" is written as a text
    that fills the image. As such, later when working with such a dataset you can
    look at the image and know which index it had before you use it.
    
    Follow this recipe to make good images:
    
    1- set n_x, n_y to 10, Set the desired n_r, n_c and width. 
    2- try fontsize that is the largest
    3- Increase n_x and n_y to desired size.
    
    You can provide a logs_root, log_dir or simply select a directory to save the
    output 4D array.
    
    """
    n_x, n_y, n_r, n_c = dataset_shape
    dataset = np.zeros((n_x, n_y, n_r, n_c))    
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    /np.log(np.maximum(n_x, n_y))) + 1
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    if(verbose):
        pBar = printprogress(n_x * n_y)
    for ind_x in range(n_x):
        for ind_y in range(n_y):
            mat = np.ones((n_r, n_c))
            number_text = number_text_base.format(ind_x = ind_x, 
                                                  ind_y = ind_y,
                                                  width = txt_width)
            fig = plt.figure(figsize = (1, 1))
            ax = fig.add_subplot(111)
            ax.imshow(mat, cmap = 'gray', vmin = 0, vmax = 1)
            ax.text(mat.shape[0]//2 - fontsize, mat.shape[1]//2 ,
                    number_text, fontsize = fontsize)
            ax.axis('off')
            buf = pltfig_to_numpy(fig)
            plt.close()
            buf2 = buf[::buf.shape[0]//n_r, ::buf.shape[1]//n_c, :3].mean(2)
            buf2 = buf2[:n_r, :n_c]
            dataset[ind_x, ind_y] = buf2.copy()
            if(verbose):
                pBar()
    return dataset