from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="calcium_clear",
    version="0.0.1a",
    install_requires=requirements,
    packages=find_packages(),
)
