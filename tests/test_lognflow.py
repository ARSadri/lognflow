#!/usr/bin/env python

"""Tests for `lognflow` package."""

import time
import pytest
import numpy as np
import matplotlib.pyplot as plt

from lognflow import lognflow, logviewer, printprogress, select_directory

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

def test_lognflow_conflict_in_names():
    logger = lognflow(temp_dir)
    logger('This is a test for conflict in names')
    logger1 = lognflow(logger.log_dir)
    logger2 = lognflow(logger.log_dir)
    logger1(logger1.log_dir)
    logger2(logger2.log_dir)

def test_log_text():
    logger = lognflow(temp_dir, print_text = False)
    logger('This is a test for log_text')    
    for _ in range(10000):
        logger(f'{_}')

    logger.log_text('not_main_script1.pdb',
           'This is a new log file for another script')
    logger.log_text('not_main_script2.test',
                    'For other log files you need to mention the log_name')
    logger.log_text('not_main_script3',
           'This is a new log file for another script', suffix = 'io')
    logger.log_text('test.to\not_main_script4.top',
                    'For other log files you need to mention the log_name')
    logger.log_text('not_main_script2',
           'This is a new log file for another script')
    logger.log_text('not_main_script2.test',
                    'For other log files you need to mention the log_name')
    for _ in range(10000):
        logger(f'{_}')

def test_logger():
    ''' test the logger call funciton
        when lognflow object is made, you can call it.
        If it is called with a string as input, it will log that into the
        main_log text file.
        If a string is given and then something else is also provided afterwards
        the funciton log_single will be called with that while the first input
        string is going to be considered as the name of a variable.
    '''
    logger = lognflow(temp_dir)
    logger('This is a test for lognflow and logger call')

    a = 20
    b = np.array([34])
    c = 'asdf'
    
    logger(a)
    logger.log_single('a', a)
    logger.log_single('aa', a, suffix = 'txt', time_tag = False)
    logger(b)
    logger.log_single('b', b)
    logger.log_single('bb', b, suffix = 'txt', time_tag = False)
    logger(c)
    logger.log_single('test/c', c, suffix = 'txt')

def test_log_flush_period():
    logger = lognflow(temp_dir, log_flush_period = 30)
    logger('This is a test for lognflow and log_var')    
    
    time_time = time.time()
    for _ in range(20):
        while(time.time() < time_time + 10):
            pass
        time_time = time.time()
        logger(f'Log{_}'*20)
        

    logger.log_text('not_main_script',
           'This is a new log file for another script')
    logger.log_text('not_main_script',
                    'For other log files you need to mention the log_name')

def test_log_var():
    logger = lognflow(temp_dir)
    logger('This is a test for lognflow and log_var')    

    for _ in range(1000):
        logger.log_var('vars/vec/v.to.txt', np.random.rand(10000))
        
def test_log_var_without_time_stamp():
    logger = lognflow(temp_dir)
    logger('This is a test for lognflow and log_var')    

    for _ in range(10):
        logger.log_single('vars/vec/v', np.random.rand(10000), 
                       time_tag = False)
        
def test_log_animation():
    var1 = np.random.rand(32, 100, 100)
    logger = lognflow(temp_dir)
    logger('This is a test for log_animation')    
    logger.log_animation('var1',var1)

def test_log_single():
    var1 = np.random.rand(100)
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_single')    
    logger.log_single('var1/var1.txt', var1)
    logger.log_single('var1/var1.npy', var1)
    a_dict = dict({'str_var': 'This is a string',
                   'var1': var1})
    logger.log_single('a_dict', a_dict)
    logger.log_single('a_dict.txt', a_dict)
    logger.log_single('a_dict2', a_dict, suffix = 'txt')

def test_log_plot():
    var1 = np.random.rand(100)
    var2 = 3 + np.random.rand(100)
    var3 = 6 + np.random.rand(100)
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_plot')    
    
    logger.log_plot(parameter_name = 'var1', 
                    parameter_value_list = var1)
    
    logger.log_plot(parameter_name = 'vars', 
                    parameter_value_list = [var1, var2, var3])
    
def test_log_hist():
    var1 = np.random.rand(10000)
    var2 = 3 + np.random.rand(10000)
    var3 = 6 + np.random.rand(10000)
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_hist')    
    
    logger.log_hist(parameter_name = 'var1', 
                    parameter_value_list = var1,
                    n_bins = 100)
    
    logger.log_hist(parameter_name = 'vars', 
                    parameter_value_list = [var1, var2, var3],
                    n_bins = 100)
    
def test_log_scatter3():
    var1 = np.random.rand(100)
    var2 = 3 + np.random.rand(100)
    var3 = 6 + np.random.rand(100)

    var3d = np.array([var1, var2, var3])
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_scatter3')    
    
    logger.log_scatter3('var3d', var3d)    
    
def test_log_plt():
    plt.imshow(np.random.rand(100, 100))
    logger = lognflow(temp_dir)
    logger('This is a test for log_plt')    
    logger.log_plt('var3d')        
    
def test_log_hexbin():
    var1 = np.random.randn(10000)
    var2 = 3 + np.random.randn(10000)

    logger = lognflow(temp_dir)
    logger('This is a test for log_hexbin')    
    
    logger.log_hexbin('hexbin', [var1, var2])    

def test_log_imshow():
    logger = lognflow(temp_dir)
    logger('This is a test for log_imshow')    
    logger.log_imshow('var3d', np.random.rand(100, 100))    

def test_log_surface():
    logger = lognflow(temp_dir)
    logger('This is a test for log_surface')    
    logger.log_surface('var3d', np.random.rand(100, 100))    

def test_prepare_stack_of_images():
    """ 
        When we have 8 set of images, each is 100x100 and there are 9 of them 
        which will appear as 3x3 tile.
    """
    stack_1 = np.random.rand(8, 100, 100, 9)
    stack_1[:, :1, :, :] = np.nan
    stack_1[:, :, :1, :] = np.nan
    stack_1[:, -1:, :, :] = np.nan
    stack_1[:, :, -1:, :] = np.nan
    stack_2 = np.random.rand(8, 100, 100)
    stack_3 = np.random.rand(8, 100, 100, 3, 9)
    stack_3[:, :1] = np.nan
    stack_3[:, :, :1] = np.nan
    stack_3[:, -1:] = np.nan
    stack_3[:, :, -1:] = np.nan
    
    list_of_stacks123 = [stack_1, stack_2, stack_3]
    logger = lognflow(temp_dir)
    logger('This is a test for prepare_stack_of_images')
    list_of_stacks123 = logger.prepare_stack_of_images(list_of_stacks123)
    logger.log_canvas('canvas_before_handling', list_of_stacks123)
    
    stack_4 = np.random.rand(1, 32, 32, 16)
    l_stack4 = logger.prepare_stack_of_images(stack_4)
    logger.log_canvas('just_one_canvas', l_stack4, cmap = 'cool')
    
def test_log_canvas():
    imgs=[]
    for _ in range(5):
        _imgs = np.random.rand(5, 100, 100)
        _imgs[:, 50, 50] = 2
        imgs.append(_imgs)
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_canvas')    
    logger(f'imgs.shape: {imgs[0].shape}')

    logger.log_single(r'test_param1', _imgs)
    logger.log_single(r'test_param2/', _imgs)
    logger.log_single(r'test_param3\\', _imgs)
    logger.log_single(r'test_param4\d', _imgs)
    logger.log_single(r'test_param4\d2\\', _imgs)
    logger.log_single(r'test_param4\d2/', _imgs)
    logger.log_single(r'test_param4\d2/e', _imgs)

    logger.log_canvas(parameter_name = 'test_canvas\\', 
                      list_of_stacks = imgs, 
                      text_as_colorbar = True)
    
def test_log_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    
    vec1 = np.random.rand(10000) > 0.8
    vec2 = np.random.rand(10000) > 0.2
    
    cm = confusion_matrix(vec1, vec2, normalize='all')
    logger = lognflow(temp_dir)
    logger('This is a test for log_confusion_matrix')
    logger.log_confusion_matrix('cm1', cm, title = 'test_log_confusion_matrix')

def test_rename():
    logger = lognflow(temp_dir)
    logger('This is a test for test_rename')
    logger.rename(logger.log_dir.name + '_new_name')
    logger('This is another test for test_rename')
    
def test_log_single_text():
    logger = lognflow(temp_dir)
    logger('This is a test for test_log_single_text', flush = True)
    var = 2
    logger.log_single('text_log\a\t/\b/\b//\\/b', 'hello\n', suffix='txt', time_tag = False)
    logger.log_single('text_log\a', 'bye\n', suffix='json', time_tag = False)
    logger.log_single('text_log\a', var, suffix='pdb', time_tag = False)
    
def test_log_imshow_complex():
    logger = lognflow(temp_dir)
    logger('This is a test for test_log_imshow_complex', flush = True)
    
    mat = np.random.rand(100, 100) + 10 * 1j * np.random.rand(100, 100)
    
    logger(f'mat is complex? {np.iscomplexobj(mat)}')
    logger.log_imshow('mat', mat)
    
def test_replace_time_with_index():
    logger = lognflow(temp_dir)
    logger('Well this is a test for logviewer')
    
    for _ in range(5):
        logger.log_single('test_param', np.array([_]))
        logger.log_single('testy/t', np.array([_]))
    
    logged = logviewer(logger.log_dir, logger)

    data_in, flist = logged.get_stack_of_files(
        'test_param', return_data=True, return_flist=True)
    
    logger(flist)

    logger.logged.replace_time_with_index('test_param')
    
    data_out, flist = logged.get_stack_of_files(
        'test_param', return_data=True, return_flist=True)
    
    logger(flist)
    
    logger(data_in)
    logger(data_out)
    
if __name__ == '__main__':
    
    #-----IF RUN BY PYTHON------#
    temp_dir = select_directory()
    #---------------------------#
    test_log_single()
    test_log_var()
    test_log_text()
    test_log_single_text()
    test_log_imshow_complex()
    test_log_imshow()
    test_log_surface()
    test_lognflow_conflict_in_names()
    test_rename()
    test_log_plot()
    test_prepare_stack_of_images()
    test_logger()
    test_log_flush_period()
    test_log_var_without_time_stamp()
    test_log_animation()
    test_log_hist()
    test_log_scatter3()
    test_log_plt()
    test_log_hexbin()
    test_log_canvas()
    test_log_confusion_matrix()
    test_replace_time_with_index()
