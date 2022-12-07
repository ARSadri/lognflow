#!/usr/bin/env python

"""Tests for `lognflow` package."""

import pytest

from lognflow import lognflow
from lognflow import logviewer
from lognflow import printprogress

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

def test_lognflow():
    temp_dir = 'c:\Alireza\logs'
    logger = lognflow(temp_dir)
    logger('Well this is my first easy log')    
    for _ in range(10000):
        logger(f'Log{_}'*200)

    for _ in range(1000):
        logger.log_var('vars/vec/v', np.random.rand(10000))

def test_logviewer():
    from PyQt5 import QtWidgets
    dialog = QtWidgets.QFileDialog()
    log_dir = dialog.getExistingDirectory(None, 'Select Folder')
    logged = logviewer(log_dir)
    print(logged.get_main_log_text())

def test_printprogress():
    N = 10000000
    pprog = printprogress(N)
    for _ in range(N):
        pprog()
    
    #assert input('Did it show you a progress bar? (y for yes)')=='y'
    
def test_log_plot():
    var1 = np.random.rand(100)
    var2 = 3 + np.random.rand(100)
    var3 = 6 + np.random.rand(100)
    
    temp_dir = 'c:\Alireza\logs'
    logger = lognflow(temp_dir)
    logger('Well this is my first easy log')    
    
    logger.log_plot(parameter_name = 'var1', 
                    parameter_value_list = var1)
    
    logger.log_plot(parameter_name = 'vars', 
                    parameter_value_list = [var1, var2, var3])

def test_log_canvas():
    imgs=[]
    for _ in range(5):
        _imgs = np.random.rand(5, 100, 100)
        _imgs[:, 50, 50] = 2
        imgs.append(_imgs)
    
    temp_dir = 'c:\Alireza\logs'
    logger = lognflow(temp_dir)
    logger('Well this is my first easy log')    
    logger(f'imgs.shape: {imgs[0].shape}')

    logger.log_canvas(parameter_name = 'test_canvas', 
                      list_of_stacks = imgs, 
                      text_as_colorbar = True)

if __name__ == '__main__':
    test_log_canvas()
    