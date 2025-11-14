from setuptools import setup, find_packages

setup(
    name="neural-interpretability",
    version="0.1.0",
    description="Neural network interpretability via resource functors",
    author="Faez Shakil",
    author_email="faez@example.com",
    url="https://github.com/faezs/homotopy-nn",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
