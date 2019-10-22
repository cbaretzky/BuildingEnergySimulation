from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='BuildingEnergySimulation',

    version='0.0.1',

    description='Simulate energy flows and transformation in buildings',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='git@github.com:cbaretzky/BuildingEnergySimulation.git',

    author='Clemens Baretzky',

    author_email='clemens.baretzky@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Energy Simulation :: Residential',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='sample setuptools development',

    packages=['BuildingEnergySimulation'],
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',

    install_requires=['geocoder',
        'geopy',
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'pvlib',
        'requests',
        'scipy'],

    package_data={
            'sample': ['package_data.dat'],
    },

    project_urls={
        'Bug Reports': 'https://github.com/cbaretzky/BuildingEnergySimulation/issues',
        'Source': 'https://github.com/cbaretzky/BuildingEnergySimulation',
    },
)
