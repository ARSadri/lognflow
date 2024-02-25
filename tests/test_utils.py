#!/usr/bin/env python

"""Tests for `lognflow` package."""
import pytest

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
    ...

if __name__ == '__main__':
    test_is_builtin_collection()
    test_stack_to_frame()
    test_ssh_system()