from setuptools import find_packages, setup
setup(
    name='s2mgrs',
    packages=find_packages(include=['s2mgrs']),
    version='0.1.0',
    description='get valid sentinel tiles',
    author='santiago.nullo@digifarm.io',
    license='MIT',
    install_requires=['numpy<=1.24.3'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    include_package_data=True,
)