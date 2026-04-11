from setuptools import setup, find_packages

setup(
    name="noiseremover",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.1",
        "tensorboard>=2.16.0",
        "pesq>=0.0.4",
        "gradio>=4.20.0",
        "matplotlib>=3.8.0",
        "pandas>=2.2.0",
    ],
)
