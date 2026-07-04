"""Setup configuration for the AFETSONAR package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="afetsonar",
    version="1.0.0",
    description=(
        "Drone-based Disaster Damage Assessment and Rescue Routing — "
        "Teknofest 2025"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AFETSONAR Team",
    python_requires=">=3.9",
    packages=find_packages(include=["afetsonar", "afetsonar.*"]),
    install_requires=[
        # Core ML
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=5.0.0,<6.0.0",
        "segmentation-models-pytorch>=0.3.0",
        # Image / augmentation
        "albumentations>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        # Data science
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        # Visualisation
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # GIS / routing
        "osmnx>=1.7.0",
        "networkx>=3.0",
        "shapely>=2.0.0",
        "geopandas>=0.14.0",
        "pyproj>=3.6.0",
        "rasterio>=1.3.0",
        "folium>=0.15.0",
        "branca>=0.7.0",
        # Config / utilities
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "exif": [
            "exifread>=3.0.0",
        ],
        "onnx": [
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "afetsonar-infer=scripts.inference:main",
            "afetsonar-pipeline=scripts.run_pipeline:main",
            "afetsonar-evaluate=scripts.evaluate:main",
            "afetsonar-export-onnx=scripts.export_onnx:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    license="Apache-2.0",
    keywords=[
        "disaster assessment",
        "semantic segmentation",
        "SegFormer",
        "knowledge distillation",
        "rescue routing",
        "drone imagery",
        "xBD",
    ],
)
