"""
   Copyright 2013 Jack Kelly (aka Daniel)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from setuptools import setup, find_packages, Extension
from os.path import join

setup(
    name='bayesianchangepoint',
    version='0.1',
    packages = find_packages(),
    install_requires = ['numpy', 'matplotlib'],
    description='An implementation of Adams and MacKay 2007'
                ' "Bayesian Online Changepoint Detection"'
                ' in Python.  This code is based on the beautifully commented'
                ' MATLAB implementation provided by Ryan Adams.',
    author='Jack Kelly',
    author_email='jack@jack-kelly.com',
    url='https://github.com/JackKelly/bayesianchangepoint',
    download_url = 'https://github.com/JackKelly/bayesianchangepoint/tarball'
                   '/master#egg=bayesianchangepoint-dev',
    long_description=open('README.md').read(),
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache 2.0',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='bayesian bayes changepoint'
)
