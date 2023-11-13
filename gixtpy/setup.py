from os import path
from setuptools import setup, find_packages

setup(
    author="Teddy Tortorici",
    description="Tools to assist in angle tuning grazing incidence x-ray scattering (GIWAXS or GISASXS) experiments",
    long_description="https://github.com/teddy-tort/gixtpy",
    name="gixtpy",
    version='0.1.1',
    packages=find_packages(include=['gixtpy.*']),
    install_requires=[
         "numpy",
        "scipy",
        "matplotlib",
        "tk",
        "tifffile",
        "jupyter",
    ],
    python_requires=">=3.8"
)
