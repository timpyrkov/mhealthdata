import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="mhealthdata",
    version="0.1.7",
    author="Tim Pyrkov",
    author_email="tim.pyrkov@gmail.com",
    description="Wearable health data to NumPy",
    long_description=read("README.md"),
    license = "MIT License",
    long_description_content_type="text/markdown",
    url="https://github.com/timpyrkov/mhealthdata",
    packages=find_packages(exclude=("docs")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=[
        "lxml",
        "matplotlib",
        "numpy>=1.20",
        "pandas",
        "pytz",
        "tqdm",
    ],
)

