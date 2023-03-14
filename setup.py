# Initialize setup
import os
import sys
from setuptools import setup,find_packages
here = os.path.abspath(os.path.dirname(__file__))

setup(name='gp_bo',
      version='0.1',
      description='GP based bayesian optimiation',
      url='http://github.com/syerramilli/gp-bo',
      author='Suraj Yerramilli',
      author_email='surajyerramilli@gmail.com',
      license='BSD 3-Clause License',
      packages=find_packages(),
      install_requires=['numpy','scipy','gpytorch','botorch','ConfigSpace'],
      zip_safe=False)