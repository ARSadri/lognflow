from .printprogress import printprogress
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import RangeSlider, TextBox

from mpl_toolkits.axes_grid1 import make_axes_locatable

def complex2hsv(data_complex, vmin=None, vmax=None):
    """ complex2hsv
        Routine to visualise complex array as 2D image with color conveying
        phase information
        data_complex must be a complex 2d image
    """
    sx, sy = data_complex.shape

    data_abs = np.abs(data_complex)
    if vmin is None:
        vmin = data_abs.min()
    if vmax is None:
        vmax = data_abs.max()
    sat = (data_abs - vmin) / (vmax - vmin)
    data_angle = np.angle(data_complex) % (2 * np.pi)
    hue = data_angle / (2 * np.pi)
    a, b = np.divmod(hue, 1.0)

    H = np.zeros((sx, sy, 3))
    H[:, :, 0] = b
    H[:, :, 1] = np.ones([sx, sy])
    H[:, :, 2] = sat

    return hsv_to_rgb(H), data_abs, data_angle

def complex2hsv_colorbar(
        fig_and_ax=None, vmin=0, vmax=1, 
        min_angle=0, max_angle=0, 
        fontsize=8, angle_threshold=np.pi / 18):
    
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 1000),
        np.linspace(-1, 1, 1000))
    conv, sat, _ = complex2hsv(xx + 1j * yy, vmax=1)

    # Set outside the circle to transparent
    mask = (xx ** 2 + yy ** 2) > 1
    conv_rgba = np.zeros((conv.shape[0], conv.shape[1], 4))
    conv_rgba[..., :3] = conv
    conv_rgba[..., 3] = 1.0  # Set alpha to 1 for everything
    conv_rgba[mask, 3] = 0  # Set alpha to 0 outside the circle
    conv_rgba[conv_rgba < 0] = 0
    conv_rgba[conv_rgba > 1] = 1
    conv_rgba = conv_rgba[::-1, :]
    if fig_and_ax is None:
        fig, ax = plt.subplots()
    else:
        try:
            fig, ax = fig_and_ax
        except Exception as e:
            print('fig_and_ax should be a two-tuple of (fig, ax). Use:')
            print('>>> fig, ax = plt.subplots()')
            raise e

    im = ax.imshow(conv_rgba, interpolation='nearest')  # Flip the image vertically
    ax.axis('off')

    diff = np.abs(max_angle - min_angle)
    # Draw lines at min and max angles if they are not too close
    if np.minimum(diff, 2 * np.pi - diff) > angle_threshold:
        for angle in [min_angle, max_angle]:
            x_end = 500 + np.cos(angle) * 500
            y_end = 500 - np.sin(angle) * 500
            ax.plot([500, x_end], [500, y_end], '--', color='gray')

    # Add text annotations for min and max values
    if int(vmin*100)/100 > 0:   #because we are going to show .2f
        ax.text(500, 500, f'{vmin:.2f}', 
                ha='center', va='center', fontsize=fontsize, color='white')

    # Calculate position for max value text and invert color for readability
    angle = 45 * np.pi / 180  # 45 degrees in radians
    x_max = int(np.cos(angle) * 500 + 300)
    y_max = int(np.sin(angle) * 500 - 200)

    bck_color = conv_rgba[y_max, x_max, :3]
    text_color = 1 - bck_color  # Invert color

    ax.text(x_max, y_max, f'{vmax:.2f}',
            ha='center', va='center', fontsize=fontsize, color=text_color)

    return fig, ax

def plt_colorbar(mappable, colorbar_aspect=2, colorbar_pad_fraction=0.05):
    """
    Add a colorbar to the current axis with consistent width.

    Parameters:
        mappable (AxesImage): The image to which the colorbar applies.
        colorbar_aspect (int): The aspect ratio of the colorbar width relative 
            to the axis width. Default is 2.
        colorbar_pad_fraction (float): The fraction of padding between the 
            axis and the colorbar. Default is 0.05.

    Returns:
        Colorbar: The colorbar added to the axis.
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    width = ax.get_position().width / colorbar_aspect
    cax = divider.append_axes("right", size=width, pad=colorbar_pad_fraction)
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
    def __init__(self, in_image, figsize=(12, 6), title=None, bins=None,
                 kwargs_for_imshow={}, kwargs_for_hist={}):
        if bins is not None:
            if not (bins in kwargs_for_hist):
                kwargs_for_hist['bins'] = bins
        
        # Adjust figsize to provide more space if needed
        self.fig, axs = plt.subplots(
            1, 2, figsize=figsize,
            gridspec_kw={'width_ratios': [5, 1], 'wspace': 0.1})
        self.fig.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.9)
        
        # Display the image
        self.im = axs[0].imshow(in_image, **kwargs_for_imshow)
        if title is not None:
            axs[0].set_title(title)
        axs[0].axis('off')
        
        cm = self.im.get_cmap()
        
        # Histogram
        n, bins = np.histogram(in_image.ravel(), **kwargs_for_hist)
        bin_centres = 0.5 * (bins[:-1] + bins[1:])
        axs[1].barh(
            bin_centres, n, height=(bins[1]-bins[0]),
            color=cm((bin_centres - bin_centres.min()) /
                         (bin_centres.max() - bin_centres.min())))
        axs[1].invert_xaxis()
        
        axs[1].yaxis.set_visible(True)
        axs[1].xaxis.set_visible(False)
        
        # Create textbox axes
        upper_text_ax = self.fig.add_axes([0.88, 0.85, 0.05, 0.05])
        lower_text_ax = self.fig.add_axes([0.88, 0.1, 0.05, 0.05])
        
        self.upper_text_box = TextBox(
            upper_text_ax, 'Max', initial=f'{in_image.max():.6f}')
        self.lower_text_box = TextBox(
            lower_text_ax, 'Min', initial=f'{in_image.min():.6f}')
        
        # Calculate the position for the slider
        slider_top = 0.85 - 0.02  # Bottom of the upper text box
        slider_bottom = 0.1 + 0.07  # Top of the lower text box
        slider_height = slider_top - slider_bottom  # Height between the two text boxes
        
        # Create slider axes on the right side of the histogram
        slider_ax = self.fig.add_axes(
            [0.895, slider_bottom, 0.02, slider_height], 
            facecolor='lightgoldenrodyellow')
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
        self.vmin_text = axs[1].text(
            0.5, self.slider.val[0], f'{self.slider.val[0]:.6f}',
            transform=axs[1].get_yaxis_transform(), 
            ha='right', va='bottom', color='k')
        self.vmax_text = axs[1].text(
            0.5, self.slider.val[1], f'{self.slider.val[1]:.6f}',
            transform=axs[1].get_yaxis_transform(),
            ha='right', va='top', color='k')
        
        self.slider.on_changed(self.update)
        self.lower_text_box.on_submit(self.update_from_text)
        self.upper_text_box.on_submit(self.update_from_text)
    
    def update(self, val):
        self.im.set_clim(val[0], val[1])
        self.lower_limit_line.set_ydata([val[0], val[0]])
        self.upper_limit_line.set_ydata([val[1], val[1]])
        
        # Update text annotations to reflect the new vmin and vmax
        self.vmin_text.set_position((0.5, val[0]))
        self.vmin_text.set_text(f'{val[0]:.6f}')
        self.vmax_text.set_position((0.5, val[1]))
        self.vmax_text.set_text(f'{val[1]:.6f}')
        
        # Update text boxes to reflect the new values
        self.lower_text_box.set_val(f'{val[0]:.6f}')
        self.upper_text_box.set_val(f'{val[1]:.6f}')
        
        self.fig.canvas.draw_idle()
    
    def update_from_text(self, text):
        try:
            lower_val = float(self.lower_text_box.text)
            upper_val = float(self.upper_text_box.text)
            if lower_val < upper_val:
                self.slider.set_val([lower_val, upper_val])
        except ValueError:
            pass
        
def plt_imshow(img, 
               colorbar = True, 
               remove_axis_ticks = False, 
               title = None, 
               cmap = None,
               angle_cmap = 'twilight_shifted',
               portrait = None,
               complex_type = 'abs_angle',
               **kwargs):
    if(not np.iscomplexobj(img)):
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap = cmap, **kwargs)
        if(colorbar):
            plt_colorbar(im)
        if(remove_axis_ticks):
            plt.setp(ax, xticks=[], yticks=[])
    else:
        if (cmap == 'complex') | (complex_type == 'complex'):
                # Convert complex data to RGB
            complex_image, data_abs, data_angle = complex2hsv(img)
        
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
            fig, ax = plt.subplots()
            im = ax.imshow(complex_image)
            if(remove_axis_ticks):
                plt.setp(ax, xticks=[], yticks=[])

            if(colorbar):
                # Create and plot the color disc as an inset
                fig, ax_inset = complex2hsv_colorbar(
                    (fig, ax.inset_axes([0.78, 0.08, 0.18, 0.18], 
                                        transform=ax.transAxes)),
                    vmin=vmin, vmax=vmax, min_angle=min_angle, max_angle=max_angle)
                ax_inset.patch.set_alpha(0)  # Make the background of the inset axis transparent
        else:
            fig = plt.figure()
            window = plt.get_current_fig_manager().window
            if (window.height() > window.width()) & (portrait is None):
                portrait = True
            if portrait:
                ax = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]
            else:
                ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
            
            if complex_type == 'abs_angle':
                im = ax[0].imshow(np.abs(img), cmap = cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[0].set_title('abs')    
                im = ax[1].imshow(np.angle(img), cmap = angle_cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[1].set_title('angle')
            elif complex_type == 'real_imag':
                im = ax[0].imshow(np.real(img), cmap = cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[0].set_title('real')    
                im = ax[1].imshow(np.imag(img), cmap = angle_cmap, **kwargs)
                if(colorbar):
                    plt_colorbar(im)
                ax[1].set_title('imag')
            
            if(remove_axis_ticks):
                plt.setp(ax[0], xticks=[], yticks=[])
                ax[0].xaxis.set_ticks_position('none')
                ax[0].yaxis.set_ticks_position('none')
                plt.setp(ax[1], xticks=[], yticks=[])
                ax[1].xaxis.set_ticks_position('none')
                ax[1].yaxis.set_ticks_position('none')
    if title is not None:
        fig.suptitle(title)
    return fig, ax

def plt_hist(vectors_list, fig_ax = None,
             n_bins = 10, alpha = 0.5, normalize = False, 
             labels_list = None, **kwargs):
    
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig, ax = fig_ax
    
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

def plt_scatter3(
        data_N_by_3, fig_ax = None, title = None, 
        elev_list = None, azim_list = None,
        make_animation = False, **kwargs):
    assert (len(data_N_by_3.shape)==2) & (data_N_by_3.shape[1] == 3), \
        'The first argument must be N x 3'
    if fig_ax is None:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax
    ax.scatter(data_N_by_3[:, 0], 
               data_N_by_3[:, 1], 
               data_N_by_3[:, 2], **kwargs)
    
    if title is not None:
            ax.set_title(title)

    if make_animation:
        stack = []
        if elev_list is None:
            elev_list = np.arange(0, 360, 45)
        if azim_list is None:
            azim_list = np.arange(0, 360, 45)
        for elev in elev_list:
            for azim in azim_list:
                ax.view_init(elev=elev, azim=azim)
                img = pltfig_to_numpy_3ch(fig)
                stack.append(img)
        return fig, ax, stack
    else:
        return fig, ax

def plt_surface(stack, fig_ax = None, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    n_r, n_c = stack.shape

    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = fig_ax

    X, Y = np.meshgrid(np.arange(n_r, dtype='int'), 
                       np.arange(n_c, dtype='int'))
    ax.plot_surface(X, Y, stack, **kwargs)
    return fig, ax

def pltfig_to_numpy_3ch(fig):
    """Convert a matplotlib figure to a numpy 2D array (RGB)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)  # Shape should be (height, width, 4) for RGBA
    buf = np.copy(buf)  # Ensure we have a copy, not a view
    return buf

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
            f' while the stack indexed as {ind} has length {len(stack)}.'
        assert (len(stack.shape) == 3) | (len(stack.shape) == 4), \
            f'The shape of the stack {ind} must have length 3 or 4, it has '\
            f'shape of {stack.shape}. Perhaps you wanted to have only '\
             'one set of images. If thats the case, put that single '\
             'image in a list.'

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
        stack, grid_locations=None, frame_shape=None, 
        titles=[], cmaps=[], colorbar=True, 
        colorbar_aspect=2, colorbar_pad_fraction=0.05,
        figsize=None, remove_axis_ticks=True, **kwargs):
    """
    Plots a list of 2D images at specified 2D grid_locations with titles 
    and colormaps.
    
    Parameters:
    stack (list of 2D arrays): List of 2D images to plot.
    grid_locations (list of tuples or None): List of subplot grid_locations 
        in (rows, cols, index) format or None to generate a grid.
    frame_shape (tuple of int or None): Shape of the grid (rows, cols) if 
        grid_locations is None. Default is None.
    titles (list of str): List of titles for each image.
    cmaps (list of str): List of colormaps for each image.
    colorbar (bool): Whether to add a colorbar beside each image. 
        Default is True.
    colorbar_aspect (int): Aspect ratio for the colorbars. Default is 2.
    colorbar_pad_fraction (float): Padding fraction for the colorbars. 
        Default is 0.05.
    figsize (tuple): Size of the figure.
    remove_axis_ticks (bool): Whether to remove axis ticks. Default is True.
    """
    if grid_locations is None:
        N = len(stack)
        if frame_shape is None:
            cols = int(np.ceil(np.sqrt(N)))
            rows = int(np.ceil(N / cols))
            frame_shape = (rows, cols)
        grid_locations = [(frame_shape[0], frame_shape[1], i+1) for i in range(N)]
    else:
        rows = max(loc[0] for loc in grid_locations)
        cols = max(loc[1] for loc in grid_locations)
        frame_shape = (rows, cols)
    
    if figsize is None:
        frame_shape = np.array(frame_shape)
        figsize = 10 * frame_shape / frame_shape.max()
    
    fig = plt.figure(figsize=figsize)
    
    for i, (image, location) in enumerate(zip(stack, grid_locations)):
        ax = fig.add_subplot(location[0], location[1], location[2])
        
        if 'cmap' in kwargs:
            cax = ax.imshow(image, **kwargs)
        else:
            try:
                _cmap = cmaps[i]
            except:
                _cmap = None
            cax = ax.imshow(image, cmap=_cmap, **kwargs)
        try:
            ax.set_title(titles[i])
        except:
            pass
        
        if remove_axis_ticks:
            ax.axis('off')
        
        if colorbar:
            plt_colorbar(cax, colorbar_aspect=colorbar_aspect,
                         colorbar_pad_fraction=colorbar_pad_fraction)
    
    fig.tight_layout()
    return fig, ax
