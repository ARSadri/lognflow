from .printprogress import printprogress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if not (type(vectors_list) is list):
        vectors_list = [vectors_list]
    for vec_cnt, vec in enumerate(vectors_list):
        bins, edges = np.histogram(vec, n_bins)
        if normalize:
            bins = bins / bins.max()
        ax.bar(edges[:-1], bins, 
                width =np.diff(edges).mean(), alpha=alpha)
        if labels_list is None:
            ax.plot(edges[:-1], bins, **kwargs)
        else:
            assert len(labels_list) == len(vectors_list)
            ax.plot(edges[:-1], bins, 
                     label = f'{labels_list[vec_cnt]}', **kwargs)
    return fig, ax

def plt_surface(stack, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    n_r, n_c = stack.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(n_r, dtype='int'), 
                       np.arange(n_c, dtype='int'))
    ax.plot_surface(X, Y, stack, **kwargs)
    return fig, ax
        
def pltfig_to_numpy(fig):
    """ from https://www.icare.univ-lille.fr/how-to-
                    convert-a-matplotlib-figure-to-a-numpy-array-or-a-pil-image/
    """
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.ubyte)
    buf.shape = (w, h, 4)
    return buf.sum(2)

def numbers_as_images_3D(data3D_shape: tuple,
                         fontsize: int, 
                         text_loc: tuple = None,
                         verbose: bool = True):
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
    n_x, n_r, n_c = data3D_shape
    
    if text_loc is None:
        text_loc = (n_r//2 - fontsize, n_c//2 - fontsize)
    
    dataset = np.zeros(data3D_shape)    
    txt_width = int(np.log(n_x)/np.log(n_x)) + 1
    number_text_base = '{ind_x:0{width}}}'
    if(verbose):
        pBar = printprogress(n_x)
    for ind_x in range(n_x):
        mat = np.ones((n_r, n_c))
        number_text = number_text_base.format(ind_x = ind_x, 
                                              width = txt_width)
        fig = plt.figure(figsize = (n_rr, n_cc), dpi = n_rc)
        ax = fig.add_subplot(111)
        ax.imshow(mat, cmap = 'gray', vmin = 0, vmax = 1)
        ax.text(text_loc[0], text_loc[1],
                number_text, fontsize = fontsize)
        ax.axis('off')
        buf = pltfig_to_numpy(fig)
        plt.close()
        dataset[ind_x] = buf.copy()
        if(verbose):
            pBar()
    return dataset

def numbers_as_images_4D(data4D_shape: tuple,
                         fontsize: int, 
                         text_loc: tuple = None,
                         verbose: bool = True):
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
    
    :param text__loc:
        text_loc should be a tuple of the location of bottom left corner of the
        text in the image.
    
    """
    n_x, n_y, n_r, n_c = data4D_shape

    if text_loc is None:
        text_loc = (n_r//2 - fontsize, n_c//2 - fontsize)
    
    dataset = np.zeros((n_x, n_y, n_r, n_c))    
    txt_width = int(np.log(np.maximum(n_x, n_y))
                    / np.log(np.maximum(n_x, n_y))) + 1
    number_text_base = '{ind_x:0{width}}, {ind_y:0{width}}'
    if(verbose):
        pBar = printprogress(n_x * n_y)
    for ind_x in range(n_x):
        for ind_y in range(n_y):
            mat = np.ones((n_r, n_c))
            number_text = number_text_base.format(
                ind_x = ind_x, ind_y = ind_y, width = txt_width)
            n_rc = np.minimum(n_r, n_c)
            n_rr = n_r / n_rc
            n_cc = n_c / n_rc
            fig = plt.figure(figsize = (n_rr, n_cc), dpi = n_rc)
            ax = fig.add_subplot(111)
            ax.imshow(mat, cmap = 'gray', vmin = 0, vmax = 1)
            ax.text(text_loc[0], text_loc[1], number_text, fontsize = fontsize)
            ax.axis('off')
            buf = pltfig_to_numpy(fig)
            plt.close()
            dataset[ind_x, ind_y] = buf.copy()
            if(verbose):
                pBar()
    return dataset

class plot_gaussian_gradient:
    """ Orignally developed for RobustGaussinFittingLibrary
    Plot curves by showing their average, and standard deviatoin
    by shading the area around the average according to a Gaussian that
    reduces the alpha as it gets away from the average.
    You need to init() the object then add() plots and then show() it.
    refer to the tests.py
    """
    def __init__(self, xlabel = None, ylabel = None, num_bars = 100, 
                 title = None, xmin = None, xmax = None, 
                 ymin = None, ymax = None, fontsize = 14):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.num_bars = num_bars
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        LWidth = 1
        font = {
                'weight' : 'bold',
                'size'   : fontsize}
        plt.rc('font', **font)
        params = {'legend.fontsize': 'x-large',
                 'axes.labelsize': 'x-large',
                 'axes.titlesize':'x-large',
                 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'}
        plt.rcParams.update(params)
        plt.figure(figsize=(8, 6), dpi=50)
        self.ax1 = plt.subplot(111)
    
    def addPlot(self, x, mu, std, gradient_color, label, 
                snr = 3.0, mu_color = None, general_alpha = 1,
                mu_linewidth = 1):

        for idx in range(self.num_bars-1):
            y1 = ((self.num_bars-idx)*mu + idx*(mu + snr*std))/self.num_bars
            y2 = y1 + snr*std/self.num_bars
            
            prob = np.exp(-(snr*idx/self.num_bars)**2/2)
            plt.fill_between(
                x, y1, y2, 
                color = (gradient_color + (prob*general_alpha,)), 
                edgecolor=(gradient_color + (0,)))

            y1 = ((self.num_bars-idx)*mu + idx*(mu - snr*std))/self.num_bars
            y2 = y1 - snr*std/self.num_bars
            
            plt.fill_between(
                x, y1, y2, 
                color = (gradient_color + (prob*general_alpha,)), 
                edgecolor=(gradient_color + (0,)))
        if(mu_color is None):
            mu_color = gradient_color
        plt.plot(x, mu, linewidth = mu_linewidth, color = mu_color, 
                 label = label)
        
    def show(self, show_legend = True):
        if(self.xmin is not None) & (self.xmax is not None):
            plt.xlim([self.xmin, self.xmax])
        if(self.ymin is not None) & (self.ymax is not None):
            plt.ylim([self.ymin, self.ymax])
        if(self.xlabel is not None):
            plt.xlabel(self.xlabel, weight='bold')
        if(self.ylabel is not None):
            plt.ylabel(self.ylabel, weight='bold')
        if(self.title is not None):
            plt.title(self.title)
        if(show_legend):
            plt.legend()
        plt.grid()
        
        plt.show()

def imshow_series(list_of_stacks, 
                  list_of_masks = None,
                  figsize_ratio = 1,
                  text_as_colorbar = False,
                  use_colorbar = False,
                  cmap = 'viridis',
                  list_of_titles = None,
                  ):
    """ imshow a stack of images or sets of images in a shelf
        input must be a list or array of images
        
        Each element of the list can appear as either:
        * n_im, n_r x n_c
        * n_im, n_r x  3  x 1
        * n_im, n_r x n_c x 3

        :param list_of_stacks
                list_of_stacks would include arrays iteratable by their
                first dimension.
        :param borders: float
                borders between tiles will be filled with this variable
                default: np.nan
    """
    n_stacks = len(list_of_stacks)
    if(list_of_masks is not None):
        assert len(list_of_masks) == n_stacks, \
            f'the number of masks, {len(list_of_masks)} and ' \
            + f'stacks {n_stacks} should be the same'
    
    # if list_of_titles is not None:
    #     assert len(list_of_titles) == n_stacks, \
    #         f'the number of titles, {len(list_of_titles)} and ' \
    #         + f'stacks {n_stacks} should be the same'
    
    n_imgs = list_of_stacks[0].shape[0]
    for ind, stack in enumerate(list_of_stacks):
        assert stack.shape[0] == n_imgs, \
            'All members of the given list should have same number of images.'
        assert (len(stack.shape) == 3) | (len(stack.shape) == 4), \
            f'The shape of the stack {ind} must have length 3 or 4, it has '\
            f' shape of {stack.shape}. Perhaps you wanted to have only one set of'\
            ' images. If thats the case, put that single image in a list.'
            
    fig = plt.figure(figsize = (n_imgs*figsize_ratio,n_stacks*figsize_ratio))
    gs1 = matplotlib.gridspec.GridSpec(n_stacks, n_imgs)
    if(use_colorbar):
        gs1.update(wspace=0.25, hspace=0)
    else:
        gs1.update(wspace=0.025, hspace=0) 
    
    for img_cnt in range(n_imgs):
        for stack_cnt in range(n_stacks):
            ax = plt.subplot(gs1[stack_cnt, img_cnt])
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            data_canvas = list_of_stacks[stack_cnt][img_cnt].copy()
            if(list_of_masks is not None):
                mask = list_of_masks[stack_cnt]
                if(mask is not None):
                    if(data_canvas.shape == mask.shape):
                        data_canvas[mask==0] = 0
                        data_canvas_stat = data_canvas[mask>0]
            else:
                data_canvas_stat = data_canvas.copy()
            data_canvas_stat = data_canvas_stat[
                np.isnan(data_canvas_stat) == 0]
            data_canvas_stat = data_canvas_stat[
                np.isinf(data_canvas_stat) == 0]
            vmin = data_canvas_stat.min()
            vmax = data_canvas_stat.max()
            im = ax.imshow(data_canvas, 
                            vmin = vmin, 
                            vmax = vmax,
                            cmap = cmap)
            if(text_as_colorbar):
                ax.text(data_canvas.shape[0]*0,
                         data_canvas.shape[1]*0.05,
                         f'{data_canvas.max():.6f}', 
                         color = 'yellow',
                         fontsize = 2)
                ax.text(data_canvas.shape[0]*0,
                         data_canvas.shape[1]*0.5, 
                         f'{data_canvas.mean():.6f}', 
                         color = 'yellow',
                         fontsize = 2)
                ax.text(data_canvas.shape[0]*0,
                         data_canvas.shape[1]*0.95, 
                         f'{data_canvas.min():.6f}', 
                         color = 'yellow',
                         fontsize = 2)
            if(use_colorbar):
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=1)
            ax.set_aspect('equal')
            ax.axis('off')
        
    
    
    return fig, None

def imshow_by_subplots( 
    stack,
    frame_shape = None,
    grid_locations = None,
    figsize = None,
    im_size_factor = None,
    im_sizes = None,
    colorbar = False,
    remove_axis_ticks = True,
    title = None,
    cmap = None,
    **kwargs):    
    
    stack_shape = stack.shape
    n_dims = len(stack_shape)
    
    FLAG_img_ready = False
    use_stack_to_frame = False
    if(n_dims == 2):
        FLAG_img_ready = True
    elif(n_dims == 3):
        if(stack_shape[2] != 3):
            use_stack_to_frame = True
        else:
            #warning that 3 dimensions as the last axis is RGB
            FLAG_img_ready = True
    elif(n_dims == 4):
            use_stack_to_frame = True
    
    if(use_stack_to_frame):
        FLAG_img_ready = True

    if(FLAG_img_ready):
        if(np.iscomplexobj(stack)):
            print('complex not supported in log_imshow_by_subplots')
            return
        else:                
            n_f = stack.shape[0]
            if grid_locations is None:
                if(frame_shape is None):
                    n_f_sq = int(np.ceil(n_f ** 0.5))
                    n_f_r, n_f_c = (n_f_sq, n_f_sq)
                else:
                    n_f_r, n_f_c = frame_shape
                
                fig, ax = plt.subplots(n_f_r,n_f_c)
                if(remove_axis_ticks):
                    plt.setp(ax, xticks=[], yticks=[])
                for rcnt in range(n_f_r):
                    for ccnt in range(n_f_c):
                        imcnt = ccnt + rcnt * n_f_c
                        if imcnt < n_f:
                            im = stack[imcnt]
                            im_ch = ax[rcnt, ccnt].imshow(
                                im, cmap = cmap, **kwargs)
                            if(colorbar):
                                plt_colorbar(im)
                            if(remove_axis_ticks):
                                plt.setp(ax, xticks=[], yticks=[])
            else:
                assert len(grid_locations) == n_f, \
                    f'length of grid_locations: {grid_locations.shape} should '\
                    f'be the same as number of images: {n_f}.'
                assert len(grid_locations.shape) == 2, \
                    'grid_locations should be n_f x 2, its shape is: '\
                    f'{grid_locations.shape}.'
                if background_image is not None:
                    background_image = background_image.squeeze()
                    assert len(background_image.shape) == 2, \
                        'The background image should be a 2D image, its shape ' \
                        f' is {background_image.shape}.'
                        
                if figsize is None:
                    grid_locations_r_min = grid_locations[:, 0].min()
                    grid_locations_r_max = grid_locations[:, 0].max()
                    grid_locations_c_min = grid_locations[:, 1].min()
                    grid_locations_c_max = grid_locations[:, 1].max()
                    grid_size = (grid_locations_r_max - grid_locations_r_min,
                                 grid_locations_c_max - grid_locations_c_min)
                    figsize = (2, 2 * grid_size[1]/grid_size[0])
                if im_sizes is None:
                    if im_size_factor is None:
                        im_size_factor = figsize[0] * figsize[1] / n_f
                        im_sizes = (im_size_factor, 
                                    im_size_factor * stack.shape[
                                        2]/stack.shape[1]) 
                
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0, 0, 1, 1])
                if background_image is not None:
                    ax.imshow(background_image)
                if title is not None:
                    ax.set_title(title)
                for ccnt, coords in enumerate(grid_locations):
                    pos = [coords[1], 1-coords[0], im_sizes[0], im_sizes[1]]
                    ax_local = fig.add_axes(pos)
                    im = ax_local.imshow(stack[ccnt], 
                                   cmap = cmap, **kwargs)
                    if(colorbar):
                        plt_colorbar(im)
                    if(remove_axis_ticks):
                        plt.setp(ax_local, xticks=[], yticks=[])
                        ax_local.xaxis.set_ticks_position('none')
                        ax_local.yaxis.set_ticks_position('none')
        
        return fig, ax
    else:
        self.log_text(
            self.log_name,
            f'Cannot imshow variable {parameter_name} with shape' + \
            f'{stack.shape}')
        return