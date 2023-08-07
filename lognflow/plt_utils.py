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
    vectors_list = list(vectors_list)
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