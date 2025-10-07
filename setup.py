"""
Setup script for neural_compiler package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "neural_compiler" / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="neural-compiler",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Compile type-checked neural architectures from Agda to JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-compiler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.7.5",
        "numpy>=1.24.0",
        "optax>=0.1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "jax[cuda12]>=0.4.20",
        ],
        "tpu": [
            "jax[tpu]>=0.4.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "neural-compile=neural_compiler.compiler:main",
        ],
    },
)
