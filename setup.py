from setuptools import setup

setup(
    name='cdspp-hda-py',
    version='0.0.1',    
    description='Python implementation of Cross Domain Structure Preserving Projection for Heterogeneous Domain Adaptation',
    url='https://github.com/iron-lion/cdspp-hda-python',
    author='Youngjun Park',
    author_email='youngjun.park.bio@gmail.com',
    license='BSD 2-clause',
    packages=['cdspp'],
    install_requires=['scipy',
                      'numpy',
                      ],
    python_requires='>=3.6',

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
