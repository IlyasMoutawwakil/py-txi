from setuptools import setup, find_packages

setup(
    name="py-tgi",
    version="0.1",
    packages=find_packages(),
    install_requires=["docker", "huggingface-hub"],
)
