#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest

from lognflow import lognflow, select_directory, logviewer, printprogress

import numpy as np

import tempfile
temp_dir = tempfile.gettempdir()

def test_printprogress():
    for N in list([2, 4, 8, 10, 20, 200, 2000, 20000, 20000000]):
        pprog = printprogress(N)
        for _ in range(N):
            pprog()

def test_printprogress_with_logger():
    logger = lognflow(temp_dir)
    N = 1500000
    pprog = printprogress(N, print_function = logger, log_time_stamp = False)
    for _ in range(N):
        pprog()
        
def test_printprogress_ETA():
    logger = lognflow(temp_dir)
    N = 500000
    pprog = printprogress(N, print_function = None)
    for _ in range(N):
        ETA = pprog()
        print(ETA)
    
def test_specific_timing():
    import time
    logger = lognflow(temp_dir)
    N = 7812
    pprog = printprogress(N, title='Inference of 7812 points. ')
    for _ in range(N):
        counter = 0
        while counter < 15000: 
            counter += 1
        pprog()

def test_generator_type():
    vec = np.arange(100)
    sum = 0
    for _ in printprogress(vec):
        sum += _
    print(f'sum: {sum}')

if __name__ == '__main__':
    #-----IF RUN BY PYTHON------#
    temp_dir = select_directory()
    #---------------------------#
    test_generator_type()
    exit()
    test_printprogress()
    test_printprogress_ETA()
    test_specific_timing()
    test_printprogress_with_logger()
    exit()    