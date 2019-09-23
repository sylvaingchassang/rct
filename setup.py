import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xdesign",
    version="0.0.1",
    author="Sylvain Chassang",
    author_email="sylvain.chassang@gmail.com",
    description="design balanced randomized experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sylvaingchassang/xdesign",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
