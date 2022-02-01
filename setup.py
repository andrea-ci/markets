#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import path as os_path
from setuptools import setup, find_packages

# Package meta-data.
NAME = 'markets'
DESCRIPTION = 'Tools for analysis of markets.'
URL = 'https://github.com/andrea-ci/markets'
AUTHOR = 'Andrea Capitanelli'
EMAIL = 'andrea.capitanelli@gmail.com'
VERSION = '0.0.1'

# Short/long description.
here = os_path.abspath(os_path.dirname(__file__))
try:
    with open(os_path.join(here, 'README.md'), 'r', encoding = 'utf-8') as file:
        long_desc = '\n' + file.read()
except FileNotFoundError:
    long_desc = DESCRIPTION

setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    author = AUTHOR,
    author_email = EMAIL,
    maintainer = AUTHOR,
    maintainer_email = EMAIL,
    url = URL,
    python_requires = '>=3.7.0',
    packages = find_packages(),
    install_requires = [
        'numpy',
        'scipy',
        'tqdm'
    ],
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    keywords = 'financial markets analysis statistics',
    license = 'MIT',
    classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    include_package_data = True,
    package_data={'': ['data/*.csv']}
)
