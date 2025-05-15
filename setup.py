from setuptools import find_packages, setup

setup(
    name="SceneSenseServer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "requests",
        "open3d-python",
        "natsort",
        "numpy",
        "scipy",
        "diffusers",
        "torch",
        "tqdm",
        "wandb",
        "huggingface-hub",
        "spconv-cu117",  # You may need to adjust CUDA version
        "opencv-python",
        "clean-fid",
    ],
    authors="Alec Reed, Lorin Achey and, Brendan Crowe",
    description="A server for  diffusion-based completion occupancy completion and frontier extension",
    python_requires=">=3.8",
)
