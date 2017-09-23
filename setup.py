from setuptools import setup
from Cython.Build import cythonize

setup(
    name='SwellPy',
    version='0.1',
    setup_require=[
        'cython>=0.26',
        'nose>=1.0'
    ],
    install_requires=[
        'scipy>=0.18',
        'numpy>=1.11',
        'peakutils>=1.0',
        'matplotlib>=2.0'
    ],
    packages=['swellpy'],
    ext_module = cythonize("*/*.pyx"),
    test_suite='nose.collector',
    tests_require=[
        'nose'
    ]
    
)