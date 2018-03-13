from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules= [
    Extension("leaky_nav_functions", 
    ['leaky_nav_functions.pyx'], 
    extra_compile_args = ['-fopenmp'], 
    extra_link_args=['-fopenmp'],
    )
]


setup(name='parallel_nav', ext_modules = cythonize(ext_modules), )
