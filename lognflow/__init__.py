"""Top-level package for lognflow."""

__author__ = 'Alireza Sadri'
__email__ = 'arsadri@gmail.com'
__version__ = '0.15.02'

from .lognflow import getLogger
from .printprogress import printprogress
from .plt_utils import (imshow_gui_by_function,
    plt_imshow, plt_imshow_subplots, subplots_grid, plt_plot, question_dialog)
from .utils import (select_directory, select_file, printv,
                    print_line, print_box, print_table, has_len, )
from .multiprocessor import multiprocessor

lognflow = getLogger

def basicConfig(*args, **kwargs):
    ...