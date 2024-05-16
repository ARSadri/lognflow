from .printprogress import printprogress
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import RangeSlider, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

def complex2hsv(data_complex, vmin = None, vmax = None):
    # Routine to visualise complex array as 2D image with colour conveying
    # phase information

    sy, sx = data_complex.shape

    sat = np.abs(data_complex)
    if vmin is None:
        vmin = sat.min()
    if vmax is None:        
        vmax = sat.max()
    sat = (sat-vmin)/(vmax-vmin)
    
    hue = (np.angle(data_complex)+np.pi)/(2*np.pi)
    a, b = np.divmod(hue,1.0)
    
    H = np.zeros((sx, sy, 3))
    H[:,:,0] = b
    H[:,:,1] = np.ones([sx, sy])
    H[:,:,2] = sat
    
    return hsv_to_rgb(H)

def complex2hsv_colorbar():
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 1000), 
        np.linspace(-1, 1, 1000))    
    conv = complex2hsv(xx + 1j*yy, vmax = 1)

    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 1000), 
        np.linspace(-1, 1, 1000))
    conv[(xx**2 + yy **2)>1] = 1
    
    fig, ax = plt.subplots()
    im = ax.imshow(conv)
    ax.axis('off')
    return fig, ax

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

def plt_violinplot(
        dataset:list, positions, facecolor = None, edgecolor = None, 
        alpha = 0.5, label = None, fig_and_ax : tuple = None, 
        plt_violinplot_kwargs = {}):
    
    if(fig_and_ax is None):
        fig, ax = plt.subplots(1)
    else:
        fig, ax = fig_and_ax
    violin_parts = ax.violinplot(dataset, positions, **plt_violinplot_kwargs)
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians','bodies'):
        vp = violin_parts.get(partname, [])
        if partname == 'bodies':
            for vp_body in vp:
                vp_body.set_facecolor(facecolor)
                vp_body.set_edgecolor(edgecolor)
                vp_body.set_alpha(alpha)
        else:
            if isinstance(vp, list):
                for v in vp:
                    v.set_edgecolor(facecolor)
            else:
                vp.set_edgecolor(facecolor)
    return fig, ax

class plt_imhist:
    def __init__(self, in_image, figsize=(12, 6), title = None,
                 kwargs_for_imshow = {}, kwargs_for_hist = {}):
        # Adjust figsize to provide more space if needed
        self.fig, axs = plt.subplots(
            1, 2, figsize = figsize,
            gridspec_kw={'width_ratios': [5, 1], 'wspace': 0.1})
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)  
        # Leave space on the right for the slider
        
        # Display the image
        self.im = axs[0].imshow(in_image, **kwargs_for_imshow)
        if title is not None:
            axs[0].set_title(title)
        axs[0].axis('off')  # Hide axes for the image
        
        cm = self.im.get_cmap()
        
        # Histogram
        n, bins = np.histogram(in_image.ravel(), **kwargs_for_hist)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        axs[1].barh(
            bin_centres, n, height=(bins[1]-bins[0]),
            color=cm((bin_centres - bin_centres.min()) /
                         (bin_centres.max() - bin_centres.min())))
        axs[1].invert_xaxis()  # Invert x-axis to have the histogram vertical
        
        axs[1].yaxis.set_visible(True)  # Make sure the y-axis is visible
        axs[1].xaxis.set_visible(False)  # Hide the x-axis
        
        # Create slider axes on the right side of the histogram
        slider_ax = self.fig.add_axes(
            [0.88, 0.1, 0.02, 0.8], facecolor='lightgoldenrodyellow')
        self.slider = RangeSlider(
            slider_ax, '', in_image.min(), in_image.max(),
            valinit=[in_image.min(), in_image.max()], orientation='vertical')
        self.slider.label.set_visible(False)
        self.slider.valtext.set_visible(False)  
        
        self.lower_limit_line = axs[1].axhline(
            self.slider.val[0], color='k', linestyle='--')
        self.upper_limit_line = axs[1].axhline(
            self.slider.val[1], color='k', linestyle='--')

        # Initial text annotations for vmin and vmax
        self.vmin_text = axs[1].text(0.5, self.slider.val[0], f'{self.slider.val[0]:.2f}',
                                     transform=axs[1].get_yaxis_transform(),
                                     ha='right', va='bottom', color='k')
        self.vmax_text = axs[1].text(0.5, self.slider.val[1], f'{self.slider.val[1]:.2f}',
                                     transform=axs[1].get_yaxis_transform(),
                                     ha='right', va='top', color='k')

        self.slider.on_changed(self.update)
        
    def update(self, val):
        self.im.set_clim(val[0], val[1])
        self.lower_limit_line.set_ydata([val[0], val[0]])
        self.upper_limit_line.set_ydata([val[1], val[1]])
        
        # Update text annotations to reflect the new vmin and vmax
        self.vmin_text.set_position((0.5, val[0]))
        self.vmin_text.set_text(f'{val[0]:.2f}')
        self.vmax_text.set_position((0.5, val[1]))
        self.vmax_text.set_text(f'{val[1]:.2f}')
        
        self.fig.canvas.draw_idle()
        
def plt_imshow(img, 
               colorbar = True, 
               remove_axis_ticks = False, 
               title = None, 
               cmap = None,
               **kwargs):
    if(not np.iscomplexobj(img)):
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap = cmap, **kwargs)
        if(colorbar):
            plt_colorbar(im)
        if(remove_axis_ticks):
            plt.setp(ax, xticks=[], yticks=[])
    else:
        if cmap == 'complex':
            converted = complex2hsv(img)
            fig, ax = plt.subplots()
            im = ax.imshow(converted, cmap = 'hsv', **kwargs)
            if(colorbar):
                plt_colorbar(im)
                
            if(remove_axis_ticks):
                plt.setp(ax, xticks=[], yticks=[])
            
        else:
            fig, ax = plt.subplots(1, 2)
            im = ax[0].imshow(np.abs(img), 
                              cmap = cmap, **kwargs)
            if(colorbar):
                plt_colorbar(im)
            ax[0].set_title('Amplitude')    
            im = ax[1].imshow(np.angle(img), 
                              cmap = cmap, **kwargs)
            if(colorbar):
                plt_colorbar(im)
            ax[1].set_title('Phase')
            if(remove_axis_ticks):
                plt.setp(ax[0], xticks=[], yticks=[])
                ax[0].xaxis.set_ticks_position('none')
                ax[0].yaxis.set_ticks_position('none')
                plt.setp(ax[1], xticks=[], yticks=[])
                ax[1].xaxis.set_ticks_position('none')
                ax[1].yaxis.set_ticks_position('none')
    if title is not None:
        ax.set_title(title)
    return fig, ax

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
        
    def __call__(self, *args, **kwargs):
        self.addPlot(*args, **kwargs)

def imshow_series(list_of_stacks, 
                  list_of_masks = None,
                  figsize = None,
                  figsize_ratio = 1,
                  text_as_colorbar = False,
                  colorbar = False,
                  cmap = 'viridis',
                  list_of_titles_columns = None,
                  list_of_titles_rows = None,
                  fontsize = None,
                  transpose = False,
                  ):
    """ imshow a stack of images or sets of images in a shelf,
        input must be a list or array of images
        
        Each element of the list can appear as either:
        * n_im, n_r x n_c
        * n_im, n_r x  3  x 1
        * n_im, n_r x n_c x 3

        :param list_of_stacks
                list_of_stacks would include arrays iterable by their
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
     
    n_imgs = list_of_stacks[0].shape[0]
    for ind, stack in enumerate(list_of_stacks):
        assert stack.shape[0] == n_imgs, \
            'All members of the given list should have same number of images.' \
            + f' while the stack indexed as {ind} has length {len(stack)}'
        assert (len(stack.shape) == 3) | (len(stack.shape) == 4), \
            f'The shape of the stack {ind} must have length 3 or 4, it has '\
            f' shape of {stack.shape}. Perhaps you wanted to have only one set of'\
            ' images. If thats the case, put that single image in a list.'

    if (list_of_titles_columns is not None):
        assert len(list_of_titles_columns) == n_stacks, \
            'len(list_of_titles_columns) should be len(list_of_stacks)' \
            + f' but it is {len(list_of_titles_columns)}.'
    if (list_of_titles_rows is not None):
        assert len(list_of_titles_rows) == n_imgs, \
            'len(list_of_titles_rows) should be len(list_of_stacks[0])' \
            + f' but it is {len(list_of_titles_rows)}.'
            
    if figsize is None:
        figsize = (n_imgs*figsize_ratio,n_stacks*figsize_ratio)
        if transpose:
            figsize = (n_stacks*figsize_ratio,n_imgs*figsize_ratio)
    if fontsize is None:
        fontsize = int(max(figsize)/4)
    
    fig = plt.figure(figsize = figsize)
    if transpose:
        gs1 = matplotlib.gridspec.GridSpec(n_stacks, n_imgs)
    else:
        gs1 = matplotlib.gridspec.GridSpec(n_imgs, n_stacks)
    if(colorbar):
        gs1.update(wspace=0.25, hspace=0)
    else:
        gs1.update(wspace=0.025, hspace=0) 
    
    for img_cnt in range(n_imgs):
        for stack_cnt in range(n_stacks):
            if transpose:
                ax = plt.subplot(gs1[stack_cnt, img_cnt])
            else:
                ax = plt.subplot(gs1[img_cnt, stack_cnt])
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
                         fontsize = fontsize)
                ax.text(data_canvas.shape[0]*0,
                         data_canvas.shape[1]*0.5, 
                         f'{data_canvas.mean():.6f}', 
                         color = 'yellow',
                         fontsize = fontsize)
                ax.text(data_canvas.shape[0]*0,
                         data_canvas.shape[1]*0.95, 
                         f'{data_canvas.min():.6f}', 
                         color = 'yellow',
                         fontsize = fontsize)
            ax.set_aspect('equal')
            if (list_of_titles_columns is not None):
                if img_cnt == 0:
                    ax.set_title(list_of_titles_columns[stack_cnt])
            if (list_of_titles_rows is not None):
                if stack_cnt == 0:
                    ax.set_ylabel(list_of_titles_rows[img_cnt])
            if (img_cnt > 0) & (stack_cnt > 0):
                ax.axis('off')
            if(colorbar):
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=1)
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
                            im = ax[rcnt, ccnt].imshow(
                                stack[imcnt], cmap = cmap, **kwargs)
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