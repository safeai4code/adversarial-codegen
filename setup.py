from setuptools import setup, find_packages
import ssl
import nltk

# Sometimes NLTK downloads fail due to SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data packages."""
    nltk_data = [
        'punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords', 
        'punkt_tab', 'averaged_perceptron_tagger_eng',
    ]
    for package in nltk_data:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

# Download NLTK data during setup
download_nltk_data()

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
    'nltk>=3.6.0',
    'evalplus',
    'attrs>=21.2.0',
    'certifi>=2020.6.20',
    'chardet>=4.0.0',
    'click>=8.0.3',
    'cryptography>=3.4.8',
    'idna>=3.3',
    'Jinja2>=3.0.3',
    'jsonschema>=3.2.0',
    'MarkupSafe>=2.0.1',
    'PyYAML>=5.4.1',
    'requests>=2.25.1',
    'six>=1.16.0',
    'urllib3>=1.26.5'
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
        'Pillow>=9.0.1',
        'pyrsistent>=0.18.1',
        'python-dateutil>=2.8.1',
    ]
}

setup(
    name="adversarial-codegen",
    version="0.1.0",
    author="Sen Fang",
    author_email="fangsen1996@gmail.com",
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