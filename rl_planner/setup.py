#!/usr/bin/env python3

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    version='0.1.0',
    name='rl_planner',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
    ],
    package_dir={'': 'src'},
    author='Julian Kunze',
    author_email='julian-kunze@gmx.de',
)

setup(
    install_requires=[
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'seaborn',
        'tqdm',
        'tensorboardX',
        'menge_gym'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    **setup_args
)
