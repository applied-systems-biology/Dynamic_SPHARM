from setuptools import setup

setup(
    name='spharm',    # This is the name of your PyPI-package.
    version='0.1',                          # python versioneer
    url="https://asb-git.hki-jena.de/AMedyukh/SPHARM",
    author="Anna Medyukhina",
    packages=['SPHARM', 'SPHARM.classes', 'SPHARM.lib'],
    package_data={'': ['tests/data/*', 'tests/data/surfaces/*', 'tests/data/synthetic_cells/*',
                       'tests/data/track_files/LN/*',
                       'tests/data/vrml/*', 'tests/data/wrl/LN/*']},
    include_package_data=True,
    author_email='anna.medyukhina@gmail.com',
    license='BSD-3-Clause',

    install_requires=[
        'scikit-image',
        'pandas',
        'numpy',
        'seaborn',
        'scipy',
        'ddt',
        'pyshtools',
        'mayavi',
        'vtk'
      ],
 )
