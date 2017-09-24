from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())
        build_ext.run(self)

setup(
    name='SwellPy',
    version='0.1',
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.26',
        'scipy>=0.18',
        'numpy>=1.11',
        'peakutils>=1.0',
        'matplotlib>=2.0'
    ],
    packages=['swellpy'],
    cmdclass={'build_ext': BuildExt},
    ext_modules = [
        Extension(
            "crepel", 
            sources=["swellpy/crepel.pyx"]
        )
    ],
    test_suite='nose.collector',
    tests_require=[
        'nose'
    ]
    
)