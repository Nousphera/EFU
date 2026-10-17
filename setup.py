"""Editable installation of the research source tree."""

from setuptools import setup, find_packages

setup(
    name="efu",
    version="1.0.0",
    description="EFU federated learning",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.11,<3.14",
    license="Apache-2.0 AND MIT",
)
