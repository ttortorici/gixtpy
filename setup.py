from setuptools import setup, find_packages


with open("README.md", "r") as f:
    readme = f.read()

setup(
    author="Teddy Tortorici",
    author_email="edward.tortorici@colorado.edu",
    description="Tools to assist in angle tuning grazing incidence x-ray scattering (GIWAXS or GISASXS) experiments",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/teddy-tort/gixtpy",
    license="MIT License",
    name="gixtpy",
    version='0.1.5',
    packages=['gixtpy'],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tk",
        "tifffile",
    ],
    python_requires=">=3.8",
    classsifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3"
    ]
)
