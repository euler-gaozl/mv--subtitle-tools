#!/usr/bin/env python3
"""
MV字幕工具集安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mv-subtitle-tools",
    version="0.1.0",
    author="Euler",
    author_email="euler@example.com",
    description="用于检测和处理视频中混色字符的工具集",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/euler-gaozl/mv-subtitle-tools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "paddlepaddle>=2.3.0",
        "paddleocr>=2.6.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "pandas>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "train-mixed-color=train_mixed_color_classifier:main",
            "process-mixed-color=process_mixed_color_batch:main",
            "dedup-mixed-color=dedup_same_second:process_results",
        ],
    },
)
