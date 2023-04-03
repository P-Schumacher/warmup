from setuptools import find_packages, setup

setup(
    name="warmup",
    version="0.2.0",
    install_requires=["gym==0.13.0, pyyaml"],
    author="Pierre Schumacher, MPI-IS Tuebingen, Autonomous Learning",
    author_email="pierre.schumacher@tuebingen.mpg.de",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "warmup": ["param_files/*.json", "xml_files/Geometry/*.stl"]
    },
)
