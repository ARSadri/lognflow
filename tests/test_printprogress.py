#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest

from lognflow import lognflow, select_directory, logviewer, printprogress

import numpy as np

import tempfile
temp_dir = tempfile.gettempdir()

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_printprogress():
    N = 3000000
    pprog = printprogress(N)
    for _ in range(N):
        pprog()
    # assert input('Did it show you a progress bar? (y for yes)')=='y'

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

if __name__ == '__main__':
    #-----IF RUN BY PYTHON------#
    temp_dir = select_directory()
    #---------------------------#
    test_printprogress_ETA()
    test_specific_timing()
    test_printprogress_with_logger()
    test_printprogress()
    exit()