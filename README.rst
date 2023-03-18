lognflow
========


.. image:: https://img.shields.io/pypi/v/lognflow.svg
        :target: https://pypi.python.org/pypi/lognflow

.. image:: https://img.shields.io/travis/arsadri/lognflow.svg
        :target: https://travis-ci.com/arsadri/lognflow

.. image:: https://readthedocs.org/projects/lognflow/badge/?version=latest
        :target: https://lognflow.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Log and Flow tracking made easy with Python. You can install it by:

.. code-block:: python
	pip install lognflow


A simple program to use it would be similar to the following:

.. code-block:: python
	from lognflow import lognflow
	import numpy as np
	vec = np.random.rand(100)
	
	logger = lognflow('c:\\test\\')
	logger('This is a test for lognflow and log_var')
	logger.log_single('vec', vec)

The logviewer is also very useful.

.. code-block:: python
	from lognflow import logviewer
	
	logged = logviewer('c:\\test\\some_log\')
	vec = logged.get_single('vec')

The printprogress makes a pretty nice progress bar.

.. code-block:: python
	from lognflow import printprogress
	
	N = 100
	pbar = printprogress(N)
	for _ in range(N):
		# do_something()
		pbar()

In this package we use a folder on the HDD to generate files and folders in typical
formats such as numpy npy and npz, png, ... to log. A log viewer is also availble
to turn an already logged flow into variables. Obviously, it will read the folders 
and map them for you, which is something you could spend hours to do by yourself.
Also there is the nicest progress bar, that you can easily understand
and use or implement yourself when you have the time.

Looking at most logging packages online, you see that you need to spend a lot of time
learning how to use them and realizing how they work. Especially when you have to deal
with http servers and ... which will be a huge pain when working for companies
who have their own HPC. 

This is why lognflow is handy and straight forward.

Many tests are avialable in the tests directory.

* Free software: GNU General Public License v3
* Documentation: https://lognflow.readthedocs.io.

Features
--------

* lognflow puts all the logs into a directory on your pc
* lognflow makes it easy to log text or simple plots.
* logviewer makes it easy to load variables or directories
* printprogress is one of the best progress bars in Python.

Credits
^^^^^^^^

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
