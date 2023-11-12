from setuptools import setup

setup(
    name='gixspy',
    version='0.1.0',
    description='A set up tools to assist in setting up grazing incidence'
                'x-ray scattering (GIWAXS or GISASXS) experiments',
    url='https://github.com/teddy-tort/giwaxs_tune/',
    author='Edward (Teddy) C Tortorici',
    author_email='edward.tortorici@colorado.edu',
    license='MIT License',
    packages=['gixspy'],
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
