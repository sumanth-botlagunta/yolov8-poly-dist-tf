from setuptools import find_packages, setup

setup(
    name="yolov8-poly-dist-tf",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10.13",
)
