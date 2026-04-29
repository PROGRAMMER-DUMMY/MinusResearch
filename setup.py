from setuptools import setup, find_packages

setup(
    name="deep-research",
    version="1.0.0",
    description="Grounded multi-agent deep research system — cited, scored, vaulted.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "deepresearch=deep_research.cli.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
