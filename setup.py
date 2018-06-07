"""
Just a regular `setup.py` file.

Author: Nikolay Lysenko
"""


import os
from setuptools import setup, find_packages


current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='data_busters',
    version='0.1',
    description='Computational experiments on synthetic data',
    long_description=long_description,
    url='https://github.com/osa-decentralized/data_busters',
    author='Nikolay Lysenko',
    author_email='nikolay.lysenko.1992@gmail.com',
    license='MIT',
    keywords='cgan discrete',
    packages=find_packages(exclude=['docs', 'tests', 'ci']),
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'pandas', 'scipy', 'tensorflow', 'PyYAML', 'tqdm'
    ]
)
