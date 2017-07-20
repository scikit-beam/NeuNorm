#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name = "NeuNorm",
    version = "1.1.1",
    author = "Jean Bilheux",
    author_email = "bilheuxjm@ornl.gov",
    packages = find_packages(exclude=['tests', 'notebooks']),
    include_package_data = True,
    test_suite = 'tests',
    install_requires = [
        'numpy',
        'pyfits',
        'pillow',
        'pathlib',
        'astropy',
        'scipy',
    ],
    dependency_links = [
    ],
    description = "neutron normalization data",
    license = 'BSD',
    keywords = "neutron normalization imaging",
    url = "https://github.com/ornlneutronimaging/NeuNorm",
    classifiers = ['Development Status :: 3 - Alpha',
                   'Topic :: Scientific/Engineering :: Physics',
                   'Intended Audience :: Developers',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'],
)


# End of file
