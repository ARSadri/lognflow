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

def test_ssh_system():
    ...

if __name__ == '__main__':
    test_stack_to_frame()
    test_ssh_system()