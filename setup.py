from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="slippi-ai-modal",
    version="0.1.0",
    author="KB",
    author_email="kael4n@gmail.com",
    description="AI agent for Super Smash Bros. Melee using imitation learning from Slippi replays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kael4n/slippi-ai-modal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "gpu": [
            "tensorflow-gpu>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slippi-train=scripts.working.experimental:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)