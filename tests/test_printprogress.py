#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest
import inspect
from lognflow import lognflow, select_directory, logviewer, printprogress

import numpy as np
import time

import tempfile
temp_dir = tempfile.gettempdir()

def test_printprogress():
    print('Testing function', inspect.currentframe().f_code.co_name)
    for N in list([100, 200, 400, 1000]):
        pprog = printprogress(N)
        for _ in range(N):
            time.sleep(0.01)
            pprog()

def test_printprogress_with_logger():
    print('Testing function', inspect.currentframe().f_code.co_name)
    logger = lognflow(temp_dir)
    N = 1500000
    pprog = printprogress(N, print_function = logger, log_time_stamp = False)
    for _ in range(N):
        pprog()
        
def test_printprogress_ETA():
    print('Testing function', inspect.currentframe().f_code.co_name)
    logger = lognflow(temp_dir)
    N = 500000
    pprog = printprogress(N, print_function = None)
    for _ in range(N):
        ETA = pprog()
        print(f'ETA: {ETA:.2f}')
    
def test_specific_timing():
    print('Testing function', inspect.currentframe().f_code.co_name)
    logger = lognflow(temp_dir)
    N = 7812
    pprog = printprogress(N, title='Inference of 7812 points. ')
    for _ in range(N):
        counter = 0
        while counter < 15000: 
            counter += 1
        pprog()

def test_generator_type():
    print('Testing function', inspect.currentframe().f_code.co_name)
    vec = np.arange(12)
    sum = 0
    for _ in printprogress(vec):
        sum += _
        time.sleep(0.1)
    print(f'sum: {sum}')

def test_varying_periods():
    print('Testing function', inspect.currentframe().f_code.co_name)
    vec = np.arange(30)
    sum = 0
    for _ in printprogress(vec):
        sum += _
        time.sleep(np.random.rand())
    print(f'sum: {sum}')

if __name__ == '__main__':
    #-----IF RUN BY PYTHON------#
    temp_dir = select_directory()
    #---------------------------#
    test_printprogress()
    test_generator_type()
    test_printprogress_ETA()
    test_specific_timing()
    test_printprogress_with_logger()
    test_varying_periods()

