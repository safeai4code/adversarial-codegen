from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # We need to import nltk only after it's installed
        import nltk
        try:
            nltk_data = [
                'punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords', 
                'punkt_tab', 'averaged_perceptron_tagger_eng',
            ]
            for package in nltk_data:
                nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        import nltk
        try:
            nltk_data = [
                'punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords', 
                'punkt_tab', 'averaged_perceptron_tagger_eng',
            ]
            for package in nltk_data:
                nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")

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
    'urllib3>=1.26.5',
    'fire>=0.5.0',
    'bitsandbytes>=0.41.1',
    'cairosvg>=2.7.1',
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
    ],
    'quant': [
        'bitsandbytes>=0.41.1',
        'optimum>=1.16.1',
        'auto-gptq>=0.5.0',
        'autoawq>=0.1.0',
    ],
    # Full installation including all dependencies
    'all': TEST_PACKAGES + [
        'tqdm>=4.67.1',
        'psutil>=6.1.0',
        'Pillow>=9.0.1',
        'pyrsistent>=0.18.1',
        'python-dateutil>=2.8.1',
        'bitsandbytes>=0.41.1',
        'optimum>=1.16.1',
        'auto-gptq>=0.5.0',
        'autoawq>=0.1.0',
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
    setup_requires=['torch>=2.5.1'],  # Added setup_requires for torch
    extras_require=EXTRA_PACKAGES,
    include_package_data=True,
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    entry_points={
        'console_scripts': [
            'adversarial-codegen=adversarial_codegen.run:main',
            'adversarial-codegen-test=adversarial_codegen.tests.integration_test.quick_test:main',
            'adversarial-codegen-noise=adversarial_codegen.noise:main',
        ],
    },
)
