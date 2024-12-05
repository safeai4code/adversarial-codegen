from setuptools import setup, find_packages

# Core dependencies required for the project
REQUIRED_PACKAGES = [
    'torch>=2.5.1',
    'transformers>=4.46.3',
    'accelerate>=1.1.1',
    'datasets>=3.1.0',
    'peft>=0.13.2',
    'numpy>=2.1.3',
    'pandas>=2.2.3',
    'huggingface-hub>=0.26.3',
]

# Testing dependencies
TEST_PACKAGES = [
    'pytest>=8.3.3',
]

# Optional dependencies
EXTRA_PACKAGES = {
    'test': TEST_PACKAGES,
    'dev': TEST_PACKAGES + [
        'tqdm>=4.67.1',
        'psutil>=6.1.0',
    ]
}

setup(
    name="adversarial-codegen",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for testing LLM robustness under adversarial attacks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adversarial-codegen",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    include_package_data=True,
)