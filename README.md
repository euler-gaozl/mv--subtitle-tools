# MV字幕工具集

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于检测和处理视频字幕中混色字符的Python工具集。

## 当前状态

这是字幕识别与时间提取的实验项目。效果受 OCR 环境、CPU/GPU 和有限标注样本影响，当前不承诺稳定识别精度。示例统计不是完整基准测试。本次更新整理中文文档，未重新执行模型训练或视频测试。

## 🌟 功能特点

- **混色字符检测**：使用机器学习识别视频字幕中的混色字符（如蓝白渐变字符）
- **OCR文字识别**：基于PaddleOCR的字符检测与识别
- **智能去重**：对检测结果进行时间窗口去重，避免重复字符
- **批量处理**：支持批量处理视频帧图像
- **可视化分析**：提供字符颜色分布的可视化分析

## 📋 项目背景

在一些MV或视频中，字幕可能使用特殊的混色效果（如蓝白渐变）来增强视觉效果。这些混色字符通常包含重要信息，但传统OCR工具难以准确识别。本项目使用计算机视觉和机器学习技术，专门设计用于检测和识别此类混色字符。

## 🔧 安装

### 前提条件

- Python 3.7+
- OpenCV
- NumPy
- PaddlePaddle
- PaddleOCR
- scikit-learn

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/euler-gaozl/mv--subtitle-tools.git
cd mv--subtitle-tools
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装PaddleOCR（如果尚未安装）：

```bash
pip install paddlepaddle
pip install paddleocr
```

## 🚀 使用方法

### 1. 准备训练数据

将训练数据放置在以下目录结构中：

```
training_data/
  ├── mixed_color/    # 混色字符图像
  └── normal_color/   # 普通字符图像
test_data/
  ├── mixed_color/    # 测试用混色字符图像
  └── normal_color/   # 测试用普通字符图像
```

### 2. 训练混色字符分类器

```bash
python train_mixed_color_classifier.py
```

### 3. 处理视频帧图像

将视频帧图像放置在`processing_output/视频ID/`目录中，然后运行：

```bash
python process_mixed_color_batch.py --all
```

### 4. 查看结果

处理结果将保存在`processing_results/`目录中：

- `all_results.json`：完整处理结果
- `mixed_characters.txt`：提取的混色字符列表

### 5. 去重处理

对同一秒内的结果进行去重：

```bash
python dedup_same_second.py
```

去重后的结果将保存为：

- `all_results_deduped.json`
- `mixed_characters_deduped.txt`

## 📊 示例

### 混色字符检测

当前仓库尚未提供该检测示例截图。

### 去重效果

原始检测结果可能包含大量重复字符：

```
原始：4375个字符
去重后：581个字符
减少了：3794个字符
```

## 🧩 项目结构

```
mv-subtitle-tools/
├── train_mixed_color_classifier.py  # 混色字符分类器训练脚本
├── process_mixed_color_batch.py     # 混色字符检测批处理
├── dedup_same_second.py             # 同一秒内去重处理
├── ocr_utils.py                     # OCR工具模块
├── models/                          # 模型目录
├── processing_output/               # 处理输入目录
├── processing_results/              # 处理结果目录
└── test/                            # 测试数据目录
```

## 📝 文档

- [API文档](docs/API.md)
- [使用教程](docs/使用教程.md)
- [贡献指南](CONTRIBUTING.md)

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。
