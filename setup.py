from setuptools import find_packages
from distutils.core import setup

setup(name='UniFP',
      version='1.0.0',
      author='zhipy&lipy',
      packages=find_packages(),
      description='UniFP',
      install_requires=['isaacgym', 
                        'matplotlib', 
                        'numpy==1.23', 
                        'tensorboard', 
                        'mujoco==3.2.3', 
                        'pyyaml', 
                        'wandb',
                        'params_proto',
                        'pydelatin'])