import os

from setuptools import setup, find_packages

this_dir = os.path.abspath(os.path.dirname(__file__))
setup_reqs = []

with open(os.path.join(this_dir, 'requirements.txt'), 'r') as fp:
    install_reqs = [r.rstrip() for r in fp.readlines() if
                    not r.startswith(('#', 'git+'))]

with open(os.path.join(this_dir, "README.md"), "r") as fh:
    long_description = fh.read()

# replacing import with text parsing to avoid importing any other packages listed in the __init__.py
# from pytolemaic.version import version
with open("pytolemaic/version.py") as fh:
    last_line = fh.readlines()[-1]
    if 'version' not in last_line:
        raise ValueError("version number must appear the last line of the version.py file")
    version = last_line.split()[-1].strip("\"'")

print(
    "Installing the following pytolemaic packages: {}".format(find_packages(exclude=['tests', 'scripts', 'examples'])))

setup(
    name='pytolemaic',
    author='Orion Talmi',
    author_email='otalmi@gmail.com',
    description='Package for ML model analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/broundal/Pytolemaic",
    version=str(version),
    packages=find_packages(exclude=['tests', 'scripts']),
    setup_requires=setup_reqs,
    install_requires=install_reqs,
    include_package_data=True,
    platforms=['Linux'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free To Use But Restricted",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6.*',
)
