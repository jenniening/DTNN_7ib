from setuptools import find_packages, setup

setup(
    name='DTNN_7ib',
    packages=find_packages("src"),
    version='0.1.0',
    description='This is a molecular energy and ligand stability prediction model bl based on deep neural tensor networks and MMFF optimized geometries',
    author='Jianing Lu',
    license='MIT',
    package_dir={"": "src"},
)
