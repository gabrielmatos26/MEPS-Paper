###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################



from setuptools import setup, Extension
import numpy

setup(
    name='MOEQ',
    version='1.00',
    description='Multi-Objective Evolutionary Q-Learning',
    packages=['neat', 'neat/nn', 'utils', 'selection'],
    include_dirs=[numpy.get_include()],
)
