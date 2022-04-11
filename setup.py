import setuptools

setuptools.setup(
    name="bsfit",
    version="0.0.1",
    author="steeve laquitaine",
    author_email="",
    description="Bayesian modeling package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
)
