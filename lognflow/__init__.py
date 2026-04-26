"""Top-level package for lognflow."""

__author__ = 'Alireza Sadri'
__email__ = 'arsadri@gmail.com'
__version__ = '0.14.03'

from .lognflow import getLogger
from .printprogress import printprogress
from .plt_utils import (
    plt_imshow, plt_imshow_subplots, subplots_grid, plt_plot, plt_hist2)
from .utils import (select_directory, select_file, block_runner, printv,
                    print_line, print_box, print_table, has_len, sanitize_config)
from .multiprocessor import multiprocessor

lognflow = getLogger

def basicConfig(*args, **kwargs):
    ...