



import ast
from setuptools import setup
import sys

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')

# Get docstring and version without importing module
with open('deepCR/__init__.py') as f:
    mod = ast.parse(f.read())

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

__doc__ = ast.get_docstring(mod)
__version__ = mod.body[-1].value.s

setup(description=__doc__.splitlines()[1],
      version=__version__,
      include_package_data=True,
      long_description=long_description,
      long_description_content_type="text/markdown",
      setup_requires=setup_requires)
