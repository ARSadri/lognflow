"""Top-level package for lognflow."""

__author__ = 'Alireza Sadri'
__email__ = 'arsadri@gmail.com'
__version__ = '0.16.0'

from .lognflow import getLogger
from .printprogress import printprogress
from .plt_utils import (
    question_dialog, plt_imshow, plt_imshow_subplots, plt_imshow_frame, plt_plot)
from .utils import (printv, print_line, print_box, print_table, has_len,
                    nohup_job_run, nohup_job_fetch, nohup_job_finish)
from .multiprocessor import multiprocessor

lognflow = getLogger

def basicConfig(*args, **kwargs):
    ...