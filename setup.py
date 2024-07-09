from setuptools import find_packages
from setuptools import setup

setup(
    name='fbg_swdm',
    author="Gabriel Rudloff",
    author_email="gabriel.rudloff@gmail.com",
    version='1.0',
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "scikit-learn", "pandas",
                      "leap_ec==0.7.0", "lssvr @ git+https://github.com/grudloff/lssvr",
                      "scipy", "torch", "pytorch-lightning", "ofiber"],
    include_package_data=True
)