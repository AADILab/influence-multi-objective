from setuptools import setup

setup(
    name='influence_moo',
    version='0.1.0',
    description='Package for multiobjective influence code',
    url='https://github.com/AADILab/influence-multi-objective/tree/main',
    author='Everardo Gonzalez',
    author_email='gonzaeve@oregonstate.edu',
    license='BSD 2-clause',
    packages=['influence_moo'],
    install_requires=[
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
