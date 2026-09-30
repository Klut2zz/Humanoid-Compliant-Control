"""Setuptools configuration for the softmimic-gym distribution."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
EXTENSION_CONFIG = tomllib.loads((ROOT / "config" / "extension.toml").read_text(encoding="utf-8"))
PACKAGE_META = EXTENSION_CONFIG["package"]

INSTALL_REQUIRES = [
    "numpy>=1.23,<2.0",
    "torch>=2.1",
    "psutil>=5.9",
    "tomli>=2.0; python_version<'3.11'",
]

EXTRAS = {
    "isaac": [
        "isaacsim[all,extscache]==4.5.0",
        "isaaclab[isaacsim,all]==2.1.0",
    ],
    "dev": [
        "ruff>=0.5",
        "black>=24.3",
        "pytest>=7.4",
    ],
}
EXTRAS["all"] = sorted({dep for group in EXTRAS.values() for dep in group})

setup(
    name="softmimic-gym",
    version=PACKAGE_META["version"],
    description=PACKAGE_META["description"],
    long_description=(ROOT.parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author=PACKAGE_META["author"],
    maintainer=PACKAGE_META["maintainer"],
    url=PACKAGE_META["repository"],
    keywords=PACKAGE_META["keywords"],
    license="MIT",
    packages=find_packages(include=["softmimic_gym", "softmimic_gym.*"]),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
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
