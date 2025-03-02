from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lm_abstractions",
    version="0.1.0",
    author="Tom Dörr",
    author_email="",
    description="A Python library for LLM abstractions with logprob-based scoring mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tom-doerr/lm_abstractions",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "dspy-ai",
        "typing-extensions",
    ],
    entry_points={
        'console_scripts': [
            'lm-abstractions=lm_abstractions:cli_main',
        ],
    },
)
