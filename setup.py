from setuptools import setup, find_packages

setup(
    name="cmbroom",
    version="0.1.5",
    description="A Python package for blind component separation of microwave sky maps.",
    author="Alessandro Carones",
    author_email="acarones@sissa.it",
    url="https://github.com/alecarones/broom",
    project_urls={
        "Homepage": "https://github.com/alecarones/broom",
        "Source": "https://github.com/alecarones/broom",
        "Issues": "https://github.com/alecarones/broom/issues",
    },
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    python_requires=">=3.10",
    packages=find_packages(where=".",include=["broom", "broom.*"]),
    include_package_data=True,
    package_data={
        "broom.configs": ["*.yaml", "*.fits"],
        "broom.utils": ["*.yaml", "*.fits"],
    },
    install_requires=[
        "astropy>=6.0.1",    
        "numpy>=1.18.5",
        "scipy>=1.8",
        "healpy>=1.15",
        "pysm3>=3.3.2",
        "mtneedlet>=0.0.5",
        "threadpoolctl>=3.6.0",   
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
