from distutils.core import setup

setup(
    name='swellpy',
    version='0.1',
    install_requires=[
        'scipy>=0.18',
        'numpy>=1.11',
        'peakutils>=1.0',
        'matplotlib>=2.0'
    ],
    packages=['swellpy']
    
)