"""Top-level package for lognflow."""

__author__ = 'Alireza Sadri'
__email__ = 'arsadri@gmail.com'
__version__ = '0.12.15'

from .lognflow import lognflow, getLogger
from .printprogress import printprogress
from .plt_utils import plt_imshow, plt_imhist, plt_hist2
from .utils import (select_directory, 
                    select_file, 
                    is_builtin_collection, 
                    text_to_collection, 
                    printv)

from .multiprocessor import multiprocessor

def basicConfig(*args, **kwargs):
    ...