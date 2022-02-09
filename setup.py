# -*- encoding: utf-8 -*-
# Copyright © 2022, enoch2090.
import io
from setuptools import setup, find_packages


setup(name='Nyquisis',
      version='1.0.3',
      description='A simple tool automatically drops the duplicate pages in PDF files.',
      keywords='Nyquisis',
      author='Enoch2090',
      author_email='gyc990926@gmail.com',
      url='https://github.com/Enoch2090/Nyquisis',
      license='GPLv3',
      long_description=io.open('./README.md', 'r', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      platforms='any',
      zip_safe=False,
      classifiers=['Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Operating System :: OS Independent',
                   ],
      packages=find_packages(exclude=()),
      include_package_data=True,
      python_requires='>=3.7',
      entry_points={
          'console_scripts': [
              'nyquisis = nyquisis.nyquisis:main',
          ]
      },
      )
