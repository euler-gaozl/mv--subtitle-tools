# MV Subtitle Tools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Tests](https://github.com/euler-gaozl/mv-subtitle-tools/actions/workflows/python-tests.yml/badge.svg)](https://github.com/euler-gaozl/mv-subtitle-tools/actions/workflows/python-tests.yml)

A Python toolkit for detecting and processing mixed-color characters in video subtitles.

[中文文档](README_CN.md)

## 🌟 Features

- **Mixed-color Character Detection**: Uses machine learning to identify mixed-color characters (like blue-white gradient characters) in video subtitles
- **OCR Text Recognition**: High-precision character detection and recognition based on PaddleOCR
- **Intelligent Deduplication**: Time-window deduplication of detection results to avoid duplicate characters
- **Batch Processing**: Support for batch processing of video frame images
- **Visualization Analysis**: Provides visualization analysis of character color distribution

## 📋 Background

In some MVs or videos, subtitles may use special mixed-color effects (such as blue-white gradients) to enhance visual effects. These mixed-color characters often contain important information, but traditional OCR tools struggle to accurately identify them. This project uses computer vision and machine learning techniques specifically designed to detect and recognize such mixed-color characters.

## 🔧 Installation

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- PaddlePaddle
- PaddleOCR
- scikit-learn

### Installation Steps

1. Clone the repository:

```bash
git clone https://github.com/euler-gaozl/mv-subtitle-tools.git
cd mv-subtitle-tools
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install PaddleOCR (if not already installed):

```bash
pip install paddlepaddle
pip install paddleocr
```

## 🚀 Usage

### 1. Prepare Training Data

Place your training data in the following directory structure:

```
training_data/
  ├── mixed_color/    # Mixed-color character images
  └── normal_color/   # Normal character images
test_data/
  ├── mixed_color/    # Test mixed-color character images
  └── normal_color/   # Test normal character images
```

### 2. Train the Mixed-color Character Classifier

```bash
python train_mixed_color_classifier.py
```

### 3. Process Video Frame Images

Place your video frame images in the `processing_output/video_id/` directory, then run:

```bash
python process_mixed_color_batch.py --all
```

### 4. View Results

Processing results will be saved in the `processing_results/` directory:

- `all_results.json`: Complete processing results
- `mixed_characters.txt`: Extracted mixed-color character list

### 5. Deduplication Processing

Deduplicate results within the same second:

```bash
python dedup_same_second.py
```

Deduplicated results will be saved as:

- `all_results_deduped.json`
- `mixed_characters_deduped.txt`

## 📊 Examples

### Mixed-color Character Detection

![Mixed-color Character Detection Example](docs/images/mixed_color_example.png)

### Deduplication Effect

Original detection results may contain many duplicate characters:

```
Original: 4375 characters
After deduplication: 581 characters
Reduced by: 3794 characters
```

## 🧩 Project Structure

```
mv-subtitle-tools/
├── train_mixed_color_classifier.py  # Mixed-color character classifier training script
├── process_mixed_color_batch.py     # Mixed-color character detection batch processing
├── dedup_same_second.py             # Same-second deduplication processing
├── ocr_utils.py                     # OCR utility module
├── models/                          # Model directory
├── processing_output/               # Processing input directory
├── processing_results/              # Processing results directory
└── test/                            # Test data directory
```

## 📝 Documentation

- [API Documentation](docs/API.md)
- [Usage Tutorial](docs/使用教程.md)
- [Contribution Guidelines](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
