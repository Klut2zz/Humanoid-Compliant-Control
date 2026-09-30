"""Setuptools configuration for the softmimic-deploy distribution."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")

BASE_REQUIREMENTS = [
    "numpy>=1.23,<2.0",
    "scipy>=1.10",
    "torch>=2.1",
    "pyyaml>=6.0",
    "opencv-python>=4.8",
    "h5py>=3.9",
    "tabulate>=0.9",
    "pandas>=1.5",
    "tqdm>=4.65",
    "pyzmq>=25.0",
]

EXTRAS = {
    "mujoco": ["mujoco>=3.1"],
    "lcm": ["lcm>=1.5.0"],
    "augmentation": ["mink>=0.3"],
    "eval": ["seaborn>=0.12"],
    "dev": [
        "pytest>=7.4",
        "ruff>=0.5",
        "black>=24.3",
    ],
}

EXTRAS["all"] = sorted({dep for group in EXTRAS.values() for dep in group})

setup(
    name="softmimic-deploy",
    version="0.1.0",
    description="Deployment, logging, and motion augmentation utilities for SoftMimic policies.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Improbable AI Lab",
    url="https://github.com/Improbable-AI/softmimic",
    license="MIT",
    packages=find_packages(
        include=[
            "softmimic_deploy",
            "softmimic_deploy.*",
            "compliant_motion_augmentation",
            "compliant_motion_augmentation.*",
        ]
    ),
    include_package_data=True,
    install_requires=BASE_REQUIREMENTS,
    extras_require=EXTRAS,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    zip_safe=False,
)
