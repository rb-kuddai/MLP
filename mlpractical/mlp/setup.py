from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(name="My first Cython app",
      ext_modules=cythonize('pool.pyx',include_path=[numpy.get_include()]),  # accepts a glob pattern
      include_dirs = [numpy.get_include()],
      )