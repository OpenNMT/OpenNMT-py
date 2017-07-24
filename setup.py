from setuptools import setup, find_packages

setup(
    name="opennmt_py",
    version="0.1.0",
    author="Avneesh Saluja",
    author_email="avneesh.saluja@airbnb.com",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url="",
    description="OpenNMT-py Package.",
    install_requires=[
        "torch==0.1.12.post2"
    ],
    dependency_links=[
        "http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl"
    ],
)
