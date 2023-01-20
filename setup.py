#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

_version = '0.4.1'

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'matplotlib']

test_requirements = ['pytest>=3', ]

setup(
    author="Alireza Sadri",
    author_email='arsadri@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Log and Flow tracking made easy with Python",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description = readme + '\n\n' + history,
    long_description_content_type = 'text/markdown',
    include_package_data=True,
    keywords='lognflow',
    name='lognflow',
    packages=find_packages(include=['lognflow', 'lognflow.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/arsadri/lognflow',
    version=_version,
    zip_safe=False,
)
