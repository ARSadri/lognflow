========
lognflow
========


.. image:: https://img.shields.io/pypi/v/lognflow.svg
        :target: https://pypi.python.org/pypi/lognflow

.. image:: https://img.shields.io/travis/arsadri/lognflow.svg
        :target: https://travis-ci.com/arsadri/lognflow

.. image:: https://readthedocs.org/projects/lognflow/badge/?version=latest
        :target: https://lognflow.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Log and Flow tracking made easy with Python.

In this package we use a folder on the HDD to generate files and folders in typical
formats such as numpy npy and npz, png, ... to log. A log viewer is also availble
to turn an already logged flow into variables. Obviously it will read the folders 
and map them for you, which is something you could spend hours to do yourself.
Also there is the nicest progress bar in the world that you can easily understand
and work with and implement yourself when you have the time.

Lokking at most logging packages online, you see that you need to spend a lot of time
learning about them and realizing how they work. Specially when you have to deal
with http servers and ... which will be a huge pain when working for companies
who have their own HPC. 

This is why lognflow is supposed to be handy and strugt forward.


* Free software: GNU General Public License v3
* Documentation: https://lognflow.readthedocs.io.


Features
--------

* Implement log_hist
* Bring scatter3 from rgflib here

Credits
^^^^^^^^

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
