#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import lognflow
import numpy as np

def test_is_builtin_collection():

    # Test the function with various types
    test_list = [1, 2, 3]
    test_string = "hello"
    test_dict = {'a': 1, 'b': 2}
    test_set = {1, 2, 3}
    test_tuple = (1, 2, 3)
    test_array = np.array([1, 2, 3])
    
    print(lognflow.is_builtin_collection(test_list))   # Expected: True
    print(lognflow.is_builtin_collection(test_string)) # Expected: False
    print(lognflow.is_builtin_collection(test_dict))   # Expected: True
    print(lognflow.is_builtin_collection(test_set))    # Expected: True
    print(lognflow.is_builtin_collection(test_tuple))  # Expected: True
    print(lognflow.is_builtin_collection(test_array))  # Expected: False

def test_ssh_system():
    try:
        ssh = ssh_system(
            hostname = 'hostname', username = 'username', password = 'password')
        remote_dir = Path('/remote/folder/path')
        local_dir = Path('/local/folder/path')
        target_fname = 'intresting_file.log'
        ssh.monitor_and_remove(remote_dir, local_dir, target_fname)
        ssh.close_connection()
    except:
        print('SSH test not passed maybe because you did not set the credentials.')
    
def test_printv():
    test1 = np.random.rand(10000)
    lognflow.utils.printv(test1)
    test2 = 123
    lognflow.utils.printv(test2)
    test3 = [1243, 'asdf', 21]
    lognflow.utils.printv(test3)

def test_save_or_load_kernel_state():
    vec = np.random.rand(100)
    vec_orig = vec.copy()
    another_variable = "Hello"
    
    current_kernel_state = lognflow.utils.save_or_load_kernel_state()
    
    vec = vec ** 2
    another_variable = "Goodbye"
    
    lognflow.utils.save_or_load_kernel_state(current_kernel_state)
    
    assert (vec == vec_orig).all()
    assert another_variable == "Hello"
    
    print("State restored successfully!")

def test_Pyrunner():
    from lognflow.utils import Pyrunner
    Pyrunner(Path('./test_pyrunner_code.py'), logger = print)

if __name__ == '__main__':
    test_Pyrunner()
    test_printv()
    test_is_builtin_collection()
    test_ssh_system()