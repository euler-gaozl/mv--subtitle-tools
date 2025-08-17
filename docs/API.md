# API 文档

本文档提供了MV字幕工具集的API参考。

## 目录

- [OCR工具模块](#ocr工具模块)
- [混色字符分类器](#混色字符分类器)
- [批量处理模块](#批量处理模块)
- [去重处理模块](#去重处理模块)

## OCR工具模块

`ocr_utils.py` 提供了OCR相关的工具函数和类。

### OCRProcessor 类

```python
class OCRProcessor:
    """OCR处理器类，封装PaddleOCR功能"""
    
    def __init__(self, use_gpu=False, lang="ch"):
        """
        初始化OCR处理器
        
        参数:
        use_gpu: 是否使用GPU，默认为False
        lang: 语言，默认为中文
        """
        
    def detect_text(self, image):
        """
        检测图像中的文本
        
        参数:
        image: 输入图像，OpenCV格式（BGR）
        
        返回:
        OCR结果列表
        """
```

### 辅助函数

```python
def has_subtitle_text(image, ocr_processor=None):
    """
    检测图像中是否包含字幕文本
    
    参数:
    image: 输入图像，OpenCV格式（BGR）
    ocr_processor: OCR处理器实例，如果为None则创建新实例
    
    返回:
    布尔值，表示是否包含字幕文本
    """
    
def is_pure_color_text(image, threshold=10):
    """
    检测文本是否为纯色
    
    参数:
    image: 输入图像，OpenCV格式（BGR）
    threshold: 颜色方差阈值，默认为10
    
    返回:
    布尔值，表示是否为纯色文本
    """
    
def extract_text_from_frame(frame, ocr_processor=None):
    """
    从视频帧中提取文本
    
    参数:
    frame: 输入帧，OpenCV格式（BGR）
    ocr_processor: OCR处理器实例，如果为None则创建新实例
    
    返回:
    提取的文本列表
    """
```

## 混色字符分类器

`train_mixed_color_classifier.py` 提供了训练混色字符分类器的功能。

### 特征提取

```python
def extract_features(image):
    """
    从字符图像中提取特征
    
    参数:
    image: 输入图像，OpenCV格式（BGR）
    
    返回:
    特征向量，包含颜色分布、梯度和纹理特征
    """
```

### 训练函数

```python
def train_classifier(mixed_color_dir, normal_color_dir, test_size=0.2, random_state=42):
    """
    训练混色字符分类器
    
    参数:
    mixed_color_dir: 混色字符图像目录
    normal_color_dir: 普通字符图像目录
    test_size: 测试集比例，默认为0.2
    random_state: 随机种子，默认为42
    
    返回:
    训练好的分类器模型
    """
```

### 评估函数

```python
def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试标签
    
    返回:
    评估指标字典
    """
```

## 批量处理模块

`process_mixed_color_batch.py` 提供了批量处理视频帧图像的功能。

### 主要函数

```python
def process_video_frames(video_id, model_path='models/mixed_color_classifier.joblib'):
    """
    处理指定视频ID的所有帧图像
    
    参数:
    video_id: 视频ID
    model_path: 模型文件路径
    
    返回:
    处理结果字典
    """
    
def process_all_videos(model_path='models/mixed_color_classifier.joblib'):
    """
    处理所有视频的帧图像
    
    参数:
    model_path: 模型文件路径
    
    返回:
    所有处理结果的字典
    """
```

### 辅助函数

```python
def detect_mixed_color_characters(image_path, model, ocr_processor):
    """
    检测图像中的混色字符
    
    参数:
    image_path: 图像文件路径
    model: 训练好的模型
    ocr_processor: OCR处理器实例
    
    返回:
    检测到的混色字符列表
    """
```

## 去重处理模块

`dedup_same_second.py` 提供了对检测结果进行去重的功能。

### 主要函数

```python
def dedup_results(results, time_window=1.0):
    """
    对检测结果进行时间窗口去重
    
    参数:
    results: 检测结果字典
    time_window: 时间窗口大小（秒），默认为1.0
    
    返回:
    去重后的结果字典
    """
    
def process_results(input_file='processing_results/all_results.json', 
                   output_file='processing_results/all_results_deduped.json',
                   text_output='processing_results/mixed_characters_deduped.txt',
                   time_window=1.0):
    """
    处理结果文件，执行去重并保存
    
    参数:
    input_file: 输入结果文件路径
    output_file: 输出结果文件路径
    text_output: 文本输出文件路径
    time_window: 时间窗口大小（秒）
    
    返回:
    去重统计信息
    """
```

## 示例用法

### 训练混色字符分类器

```python
from train_mixed_color_classifier import train_classifier, evaluate_model

# 训练分类器
model = train_classifier('training_data/mixed_color', 'training_data/normal_color')

# 保存模型
import joblib
joblib.dump(model, 'models/mixed_color_classifier.joblib')
```

### 处理视频帧

```python
from process_mixed_color_batch import process_video_frames

# 处理特定视频的帧
results = process_video_frames('video_id')

# 输出结果
print(f"检测到 {len(results['mixed_characters'])} 个混色字符")
```

### 去重处理

```python
from dedup_same_second import process_results

# 执行去重
stats = process_results()

# 输出统计信息
print(f"原始: {stats['original']} 个字符")
print(f"去重后: {stats['deduped']} 个字符")
print(f"减少了: {stats['reduced']} 个字符")
