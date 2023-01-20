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

def test_printprogress():
    N = 10000000
    pprog = printprogress(N)
    for _ in range(N):
        pprog()
    # assert input('Did it show you a progress bar? (y for yes)')=='y'
    
if __name__ == '__main__':
    test_printprogress()