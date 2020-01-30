from __future__ import absolute_import
from setuptools import setup
from setuptools import find_packages



try:
    # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:
    # for pip <= 9.0.3
    from pip.req import parse_requirements

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    return [str(ir.req) for ir in reqs]


setup(name='elephas',
      version='0.4.2',
      description='Deep learning on Spark with Keras',
      url='http://github.com/francescogalloni94/elephas',
      download_url='https://github.com/francescogalloni94/elephas/tarball/0.4.2',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=load_requirements("requirements.txt"),
      extras_require={
        'java': ['pydl4j>=0.1.3'],
        'tests': ['pytest', 'pytest-pep8', 'pytest-cov', 'mock']
    },
      packages=find_packages(),
      license='MIT',
      zip_safe=False,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ])
