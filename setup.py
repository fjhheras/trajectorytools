from setuptools import setup

setup(name='trajectorytools',
      version='0.1',
      description='trajectorytools',
      url='http://github.com/fjhheras/trajectorytools',
      author='Francisco J.H. Heras',
      author_email='fjhheras@gmail.com',
      license='GPL',
      
      packages=['trajectorytools'],
      package_data={'trajectorytools': ['data/*.npy']},
      include_package_data=True,
      zip_safe=False)

