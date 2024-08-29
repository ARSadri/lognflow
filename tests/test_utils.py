#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import lognflow
import numpy as np

def test_stack_to_frame():
   data4d = np.random.rand(25, 32, 32, 3)
   img = lognflow.stack_to_frame(data4d, borders = np.nan)
   plt.figure()
   plt.imshow(img)
   
   data4d = np.random.rand(32, 32, 16, 16, 3)
   stack = data4d.reshape(-1, *data4d.shape[2:])
   frame = lognflow.stack_to_frame(stack, borders = np.nan)
   plt.figure()
   im = plt.imshow(frame)
   lognflow.plt_colorbar(im)
   plt.show()

def test_is_builtin_collection():

    # Test the function with various types
    test_list = [1, 2, 3]
    test_string = "hello"
    test_dict = {'a': 1, 'b': 2}
    test_set = {1, 2, 3}
    test_tuple = (1, 2, 3)
    test_array = np.array([1, 2, 3])
    
    print(lognflow.is_builtin_collection(test_list))  # Expected: True
    print(lognflow.is_builtin_collection(test_string))  # Expected: False
    print(lognflow.is_builtin_collection(test_dict))  # Expected: True
    print(lognflow.is_builtin_collection(test_set))  # Expected: True
    print(lognflow.is_builtin_collection(test_tuple))  # Expected: True
    print(lognflow.is_builtin_collection(test_array))  # Expected: False


def test_ssh_system():
    try:
        ssh = ssh_system(
            hostname = 'hostname', username = 'username', password = 'password')
        remote_dir = Path('/remote/folder/path')
        local_dir = Path('/local/folder/path')
        target_fname = 'intresting_file.log'
        ssh.monitor_and_move(remote_dir, local_dir, target_fname)
        ssh.close_connection()
    except:
        print('SSH test not passed maybe because you did not set the credentials.')
    
def test_printvar():
    test1 = np.random.rand(10000)
    lognflow.utils.printvar(test1)
    test2 = 123
    lognflow.utils.printvar(test2)
    test3 = [1243, 'sadf', 21]
    lognflow.utils.printvar(test3)

def test_save_or_load_kernel_state():
    
    # Example code that sets up some variables
    vec = np.random.rand(100)
    vec_orig = vec.copy()
    another_variable = "Hello, World!"
    
    # Save the current state of the kernel (interpreter)
    current_kernel_state = lognflow.utils.save_or_load_kernel_state()
    
    # Modify the variables
    vec = vec ** 2
    another_variable = "Goodbye, World!"
    
    # Restore the previous state
    lognflow.utils.save_or_load_kernel_state(current_kernel_state)
    
    # Check that the state has been restored
    assert (vec == vec_orig).all()
    assert another_variable == "Hello, World!"
    
    print("State restored successfully!")

def test_Pyrunner():
    from lognflow import Pyrunner
    Pyrunner(Path('./test_pyrunner_code.py'), logger = print)

if __name__ == '__main__':
    test_Pyrunner();exit()
    test_printvar()
    test_is_builtin_collection()
    test_stack_to_frame()
    test_ssh_system()
