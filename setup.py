from setuptools import setup, find_packages


setup(
  name = 'detectors_eva',  # notice, there shouldn’t be ‘-’ for the pkg name('_'is accepted tho)!!
  packages = find_packages(),  # it will look up the avaible directory where there are __init__.py under it!
  version = '0.0.1',
  license='MIT',
  author = 'Jinjing',
)

