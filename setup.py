"""
Setup configuration for the heirs-property package.
"""

from setuptools import setup, find_packages

setup(
    name="heirs-property",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml>=5.1",
        "pandas>=1.0.0",
        "geopandas>=0.9.0",
        "rasterio>=1.2.0",
        "shapely>=1.7.0",
        "pytest>=6.0.0"
    ],
    python_requires=">=3.8",
) 