#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest

import time
import numpy as np
import matplotlib.pyplot as plt
from lognflow import (
    lognflow, printprogress, select_directory, text_to_object)
from lognflow.utils import stacks_to_frames

import tempfile
temp_dir = tempfile.gettempdir()

def test_log_index():
    logger = lognflow(temp_dir)
    logger('This is a test for putting index instead of time_stamp')
    logger.log_single('testa', 'testa', time_tag = False)
    logger.log_single('testb', 'testb', time_tag = True)
    logger.log_single('testb', 'test3', time_tag = True)
    logger.log_single('testc', 'test4', time_tag = 'index')
    logger.log_single('testc', 'test5', time_tag = 'index')
    logger.log_single('testc', 'test6', time_tag = 'index')
    logger.log_single('testd', 'test7', time_tag = 'time_and_index')
    logger.log_single('testd', 'test8', time_tag = 'time_and_index')
    logger.log_single('testd', 'test9', time_tag = 'time_and_index')
    
    
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
    logger.log_text('test.to\not_main_script4.top',
                    'To see if variable names are OK when passed properly')
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
    
    logger.log_scatter3('var3d', var3d.T)
    logger.log_scatter3('var3d_animation', var3d.T, make_animation=True)
    
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
    logger.log_imshow('var2d', np.random.rand(100, 100))    
    logger.log_imshow('var2d_100_of_them', np.random.rand(25, 100, 100))

def test_log_surface():
    logger = lognflow(temp_dir)
    logger('This is a test for log_surface')    
    logger.log_surface('var3d', np.random.rand(100, 100))    

def test_log_imshow_series():
    """ 
        When we have 8 set of images, each is 100x100 and there are 9 of them 
        which will appear as 3x3 tile.
    """
    logger = lognflow(temp_dir)
    logger('This is a test for prepare_stack_of_images')

    stack_1 = np.random.rand(8, 9, 100, 100)
    stack_1 = stacks_to_frames(stack_1)
    stack_2 = np.random.rand(8, 100, 100)
    stack_3 = np.random.rand(8, 9, 100, 100, 3)
    stack_3 = stacks_to_frames(stack_3)
    
    list_of_stacks123 = [stack_1, stack_2, stack_3]
    
    logger.log_imshow_series('imshow_series_before_handling', list_of_stacks123)
    
    stack_4 = np.random.rand(1, 32, 32, 16)
    stack_4 = stacks_to_frames(stack_4)
    list_of_stacks = [stack_4]
    logger.log_imshow_series('just_one_series', list_of_stacks, cmap = 'cool')
    
    imgs=[]
    for _ in range(5):
        _imgs = np.random.rand(5, 100, 100)
        _imgs[:, 50, 50] = 2
        imgs.append(_imgs)
    
    logger = lognflow(temp_dir)
    logger('This is a test for log_imshow_series')    
    logger(f'imgs.shape: {imgs[0].shape}')

    logger.log_imshow_series(parameter_name = 'log_imshow_series\\', 
                             list_of_stacks = imgs, 
                             text_as_colorbar = True)

def test_names_with_slashes_and_backslashes():
    logger = lognflow(temp_dir)
    logger('This is a test for test_names_with_slashes_and_backslashes')   

    _imgs = np.random.rand(10, 10)
    logger.log_single(r'test_param1', _imgs)
    logger.log_single(r'test_param2/', _imgs)
    logger.log_single(r'test_param3\\', _imgs)
    logger.log_single(r'test_param4\d', _imgs)
    logger.log_single(r'test_param4\d2\\', _imgs)
    logger.log_single(r'test_param4\d2/', _imgs)
    logger.log_single(r'test_param4\d2/e', _imgs)

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
    logger('Well this is a test for reading using logger')
    
    for _ in range(5):
        logger.log_single('test_param', np.array([_]))
        logger.log_single('testy/t', np.array([_]))
    
    flist = logger.get_flist('test_param*')
    data_in = logger.get_stack_from_files('test_param*')
    
    logger(flist)

    logger.replace_time_with_index('test_param*')
    
    flist = logger.get_flist('test_param*')
    data_out = logger.get_stack_from_files(flist = flist)
    
    logger(flist)
    
    logger(data_in)
    logger(data_out)

def test_copy_file():
    logger = lognflow(temp_dir)
    logger('Well this is a test for copying files')
    
    var = np.random.rand(10)
    fpath = logger.log_single('var', var, suffix = 'txt')
    
    logger.copy('myvar/varvar', fpath, suffix = 'pdb', 
             time_tag= True)
    
    var_check = logger.get_single('myvar/varvar*')
    assert str(var) == var_check
    
def test_copy_list_of_files():
    logger = lognflow(temp_dir)
    logger('Well this is a test for copy list of files')
    
    logger.log_single('test/var', np.random.rand(10), suffix = 'txt')
    logger.log_single('test/var', np.random.rand(10), suffix = 'fasta')
    logger.log_single('test/var', np.random.rand(10), suffix = 'json')
    logger.log_single('test/var', np.random.rand(10), suffix = 'txt')
    
    logger.copy('myvar/', 'test/var*', suffix = 'pdb', time_tag = False)
    
    for test_cnt in range(4):
        var_check1 = logger.get_single('test/var', file_index = test_cnt)
        var_check2 = logger.get_single('myvar/var', file_index = test_cnt)
        assert var_check1 == var_check2
    
def test_log_imshow_by_subplots():
    logger = lognflow(temp_dir)
    logger('Well this is a test for log_imshow_by_subplots')
    images = np.random.rand(20, 100, 100)
    logger.log_imshow_by_subplots('images', images, frame_shape = (4, 5))

def test_copy():
    logger1 = lognflow(temp_dir)
    fpath = logger1('Well this is a test for test_copy')
    
    logger2 = lognflow(temp_dir)
    logger2.copy('some_text.txt', fpath)

def test_log_images_to_pdf():
    logger = lognflow(temp_dir)
    logger('test log images in pdf')
    
    logger.log_imshow('im1', np.random.randn(30, 30))
    logger.log_imshow('im1', np.random.randn(20, 40))
    
    images = logger.get_stack_from_names('im1*.*')
    logger.log_images_in_pdf(
        'im1_all', parameter_value = images, time_tag = False)
    
def test_variables_to_pdf():
    logger = lognflow(temp_dir)
    logger('test log variables in pdf')
    
    logger.log_imshow('im1', np.random.randn(30, 30))
    logger.log_imshow('im1', np.random.randn(20, 40))
    logger.variables_to_pdf('im1_all', 'im1*.*', time_tag = False)

def test_log_code():
    logger = lognflow(temp_dir)
    logger('test log variables in pdf')
    
    logger.log_code(__file__)

def test_get_flist_multiple_directories():
    logger = lognflow(temp_dir)
    logger('Well this is a test for test_multiple_directories_get_flist')
    
    logger.log_single('dir1/dir/var', np.random.rand(100))
    logger.log_single('dir2/dir/var', np.random.rand(100))
    logger.log_single('dir3/dir/var', np.random.rand(100))
    
    flist = logger.get_flist('dir*/dir/var*.npy')
    [print(_) for _ in flist]
    [print(logger.name_from_file(_)) for _ in flist]
        
def test_get_stack_from_files():
    logger = lognflow(temp_dir)
    
    logger('Well this is a test for get_stack_from_files')

    for _ in range(5):
        logger.log_single('A/img', np.random.rand(100, 100))
        logger.log_single('B/img', np.random.randn(100, 100))

    flist_A = logger.get_flist('A/*')
    flist_B = logger.get_flist('B/*')
    
    logger(flist_A)
    logger(flist_B)
    
    logger.replace_time_with_index('A/img')
    logger.replace_time_with_index('B/img')

    flist_A = logger.get_flist('A/*')
    flist_B = logger.get_flist('B/*')
    
    logger(flist_A)
    logger(flist_B)
    
    stack_A = logger.get_stack_from_files(flist = flist_A)
    stack_B = logger.get_stack_from_files(flist = flist_B)

    logger(f'stack_A.shape: {stack_A.shape}')
    logger(f'stack_B.shape: {stack_B.shape}')
    
    logger.log_imshow_series('data_samples', [stack_A, stack_B], dpi = 300)

    flist_A_AB, flist_B_AB = logger.get_common_files('A/*', 'B/*')
    logger(f'flist_A_AB: {flist_A_AB}')
    logger(f'flist_B_AB: {flist_B_AB}')
    
    if(flist_A_AB):
        
        dataset_A = logger.get_stack_from_files('A/*', flist = flist_A_AB)
        dataset_B = logger.get_stack_from_files('B/*', flist = flist_B_AB)
        
        logger.log_imshow_series('data_samples', 
                                 [dataset_A, dataset_B], dpi = 300)
        _ = logger._loggers_dict['main_log.txt'].log_size
        logger('Size of the log file in bytes is: ' \
               + f'{_}')

def test_text_to_object():
    logger = lognflow(temp_dir, time_tag = False)
    test_list = ['asdf', 1243, "dd"]
    logger.log_single('test_list', test_list, suffix = 'txt')
    
    test_dict = {"one": "asdf", 'two': 1243, 'thre': "dd"}
    logger.log_single('test_dict', test_dict, suffix = 'txt')
    
    flist = logger.get_flist('*')
    print(flist)
    for file_name_input in flist:
        print('='*60)
        print(f'file name: {file_name_input}')
        with open(file_name_input, 'r') as opened_txt:
            txt = opened_txt.read()
        print('text read from the file:')
        print(txt)
        print('- '*30)
        ext_obj = text_to_object(txt)
        print(f'Extracted object is of type {type(ext_obj)}:')
        print(ext_obj)

def test_get_single_specific_fname():
    logger = lognflow(temp_dir)
    logger('test get single specific fname')
    
    vec = np.array([1])
    logger.log_single('vec', vec, time_tag = False)

    vec2 = np.array([2])
    logger.log_single('vec2', vec2, time_tag = False)
    
    vec_out = logger.get_single('vec.npy')
    
    assert vec_out == vec

def test_get_stack_from_names():
    logger = lognflow(temp_dir)
    logger('test get single specific fname')
    
    logger.log_imshow('im1', np.random.randn(30, 30))
    logger.log_imshow('im1', np.random.randn(20, 40))
    
    images = logger.get_stack_from_names('im1*')
    
    print(len(images))

def test_depricated_logviewer():
    logger = lognflow(temp_dir)
    logger('testing the depricated logviewer')
    
    logger.log_single('test', 1)
    print(logger.logged.get_single('test*'))

if __name__ == '__main__':
    #-----IF RUN BY PYTHON------#
    temp_dir = select_directory()
    #---------------------------#
    #tests about reading back
    test_log_scatter3()
    test_log_animation()
    test_log_plt()
    test_log_hexbin()
    test_log_confusion_matrix()
    test_names_with_slashes_and_backslashes()
    test_copy()
    test_log_code()
    test_get_stack_from_files()
    test_get_stack_from_names()
    test_get_flist_multiple_directories()
    test_get_single_specific_fname()
    test_text_to_object()
    test_depricated_logviewer()
    test_replace_time_with_index()
    test_log_hist()
    test_variables_to_pdf()
    test_log_images_to_pdf()
    test_copy_file()
    test_log_code()    
    test_log_index()
    test_log_imshow()
    test_copy_list_of_files()
    test_log_imshow_series()
    test_log_imshow_by_subplots()
    test_log_imshow_complex()
    test_log_var()
    test_log_text()
    test_log_single()
    test_log_single_text()
    test_log_surface()
    test_lognflow_conflict_in_names()
    test_rename()
    test_log_plot()
    test_logger()
    test_log_flush_period()
    test_log_var_without_time_stamp()
