from setuptools import setup

setup(
    name='dynamic_spharm',    # This is the name of your PyPI-package.
    version='1.0',                          # python versioneer
    url="https://github.com/applied-systems-biology/Dynamic_SPHARM/",
    author="Anna Medyukhina",
    author_email='anna.medyukhina@gmail.com',
    packages=['SPHARM', 'SPHARM.classes', 'SPHARM.lib'],
    package_data={'': ['tests/data/*', 'tests/data/surfaces/*', 'tests/data/synthetic_cells/*',
                       'tests/data/track_files/LN/*',
                       'tests/data/vrml/*', 'tests/data/wrl/LN/*']},
    include_package_data=True,
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
        'helper_lib',
        'vtk'
      ],
    dependency_links=[
        "https://github.com/applied-systems-biology/HelperLib/releases/",
    ],
 )
