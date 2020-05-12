from setuptools import setup

setup(name='trajectorytools',
      version='0.3-alpha',
      description='trajectorytools',
      url='http://github.com/fjhheras/trajectorytools',
      author='Francisco J.H. Heras',
      author_email='fjhheras@gmail.com',
      license='GPL',

      packages=['trajectorytools'],
      package_data={'trajectorytools': ['data/*.npy']},
      include_package_data=True,
      zip_safe=False,

      classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities'
      ],

      )

