#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest

from lognflow import lognflow, select_directory, logviewer, printprogress

import numpy as np

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

def test_logviewer():
    temp_dir = select_directory()
    logger = lognflow(temp_dir)
    logger('Well this is a test for logviewer')
    
    logger.log_single('test_param', np.random.rand(100))
    
    logged = logviewer(logger.log_dir, logger)
    print(logged.get_variable('test_param'))
    print(logged.get_log_text())

def test_get_images_as_stack():
    root_dir = select_directory()
    logger = lognflow(root_dir)
    
    logger('Well this is a test for logviewer')

    for _ in range(5):
        logger.log_imshow('A/', np.random.rand(100, 100), dpi = 300)
        logger.log_imshow('B/', np.random.randn(100, 100), dpi = 300)

    logged = logviewer(logger.log_dir, logger)
    stack_A = logged.get_stack_of_files('A/')
    stack_B = logged.get_stack_of_files('B/')

    print(stack_A.shape)
    print(stack_B.shape)
    
    logger.log_canvas('data_samples', [stack_A, stack_B], dpi = 300)

    flist_A = logged.get_stack_of_files(
        'A/', return_data=False, return_flist=True)
    flist_B = logged.get_stack_of_files(
        'B/', return_data=False, return_flist=True)
    
    logger(flist_A)
    logger(flist_B)

    flist_A_AB, flist_B_AB = logged.common_files('A/', 'B/')
    logger(flist_A_AB)
    logger(flist_B_AB)
    
    if(flist_A_AB):
        
        dataset_A = logged.get_stack_of_files(flist = flist_A_AB)
        dataset_B = logged.get_stack_of_files(flist = flist_B_AB)
        
        logger.log_canvas('data_samples', [dataset_A, dataset_B], dpi = 300)
        
        logger(logger._loggers_dict['main_log'][2])

def test_replace_time_with_index():
    temp_dir = select_directory()
    logger = lognflow(temp_dir)
    logger('Well this is a test for logviewer')
    
    for _ in range(5):
        logger.log_single('test_param', np.array([_]))
        logger.log_single('testy/t', np.array([_]))
    
    logged = logviewer(logger.log_dir, logger)

    data_in, flist = logged.get_stack_of_files(
        'test_param', return_data=True, return_flist=True)
    
    logger(flist)

    logged.replace_time_with_index('test_param')
    
    data_out, flist = logged.get_stack_of_files(
        'test_param', return_data=True, return_flist=True)
    
    logger(flist)
    
    logger(data_in)
    logger(data_out)
    

if __name__ == '__main__':
    test_replace_time_with_index()
    exit()
    test_get_images_as_stack()
    test_logviewer()