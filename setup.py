from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gixtpy',
    version='0.1.0',
    description='Tools to assist in angle tuning grazing incidence'
                'x-ray scattering (GIWAXS or GISASXS) experiments',
    url='https://github.com/teddy-tort/gixtpy/',
    author='Edward C Tortorici',
    author_email='edward.tortorici@colorado.edu',
    license='MIT License',
    packages=['src.gixtpy'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tk',
                      'tifffile',
                      'jupyter'],
    classifiers=[
        'Development Status :: In Development',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
