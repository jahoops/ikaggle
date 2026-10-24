from setuptools import setup, find_packages

setup(
    name="ikaggle",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'deap'
    ]
)