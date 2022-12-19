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
    
    log_dir =  select_directory(temp_dir)
    logged = logviewer(log_dir, logger)
    print(logged.get_variable('test_param'))
    print(logged.get_log_text())
    
if __name__ == '__main__':
    test_logviewer()
    
    
