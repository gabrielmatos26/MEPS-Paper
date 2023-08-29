###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################



from setuptools import setup, Extension
import numpy

setup(
    name='MEPS',
    version='2.00',
    description='Multi-Objective Evolutionary Policy Search',
    packages=['neat', 'neat/nn', 'utils', 'selection'],
    include_dirs=[numpy.get_include()],
)
