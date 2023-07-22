from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="plane_inspection",
    version="0.0.0",
    description="",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Maik Knof",
    author_email="maik.knof@gmx.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: >=3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    python_requires=">=3.8",
)
