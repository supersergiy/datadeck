import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datastacks",
    version="0.0.0.1",
    author="Sergiy Popovych",
    author_email="sergiy.popovich@gmail.com",
    description="A tool for parametrized loading from H5 datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supersergiy/datastacks",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
      'torch',
      'torchvision',
      'numpy'
    ],
    packages=setuptools.find_packages(),
)
