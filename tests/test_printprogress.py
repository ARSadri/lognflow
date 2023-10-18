#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest

from lognflow import lognflow, select_directory, logviewer, printprogress

import numpy as np
import time

import tempfile
temp_dir = tempfile.gettempdir()

def test_printprogress():
    for N in list([100, 200, 400, 1000, 2000, 4000, 6000]):
        pprog = printprogress(N)
        for _ in range(N):
            time.sleep(0.01)
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
    logger = lognflow(temp_dir)
    N = 7812
    pprog = printprogress(N, title='Inference of 7812 points. ')
    for _ in range(N):
        counter = 0
        while counter < 15000: 
            counter += 1
        pprog()

def test_generator_type():
    vec = np.arange(12)
    sum = 0
    for _ in printprogress(vec):
        sum += _
        time.sleep(0.1)
    print(f'sum: {sum}')

def test_varying_periods():
    vec = np.arange(60)
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

