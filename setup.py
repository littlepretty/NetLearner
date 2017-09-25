from setuptools import setup


setup(name='netlearner',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/littlepretty/netlearner',
      author='Jiaqi Yan',
      author_email='littlepretty881203@gmail.com',
      license='MIT',
      packages=['netlearner', 'visualization'],
      install_requires=[
          'enum',
          'tabulate',
          'matplotlib',
          'numpy',
          'sklearn',
          'tensorflow'
      ],
      zip_safe=False)
