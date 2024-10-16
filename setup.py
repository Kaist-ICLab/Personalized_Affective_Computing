from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(include=["arpreprocessing", "utils", "multimodal_classfiers", "multimodal_classfiers_finetuning", "clustering"]),
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)