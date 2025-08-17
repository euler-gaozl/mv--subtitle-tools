#!/usr/bin/env python3
"""
基本功能测试
"""

import unittest
import os
import sys
import numpy as np
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from ocr_utils import OCRProcessor, has_subtitle_text, is_pure_color_text

class TestBasicFunctions(unittest.TestCase):
    """测试基本功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建一个简单的测试图像
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # 白色背景
        # 在图像中央绘制一个蓝色文字区域
        cv2.putText(self.test_image, "测试", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    def test_ocr_processor_init(self):
        """测试OCR处理器初始化"""
        try:
            ocr_processor = OCRProcessor()
            self.assertIsNotNone(ocr_processor)
        except Exception as e:
            self.fail(f"OCR处理器初始化失败: {e}")
    
    def test_extract_features(self):
        """测试特征提取函数"""
        # 导入特征提取函数
        try:
            from train_mixed_color_classifier import extract_features
            
            # 提取特征
            features = extract_features(self.test_image)
            
            # 检查特征向量
            self.assertIsInstance(features, np.ndarray)
            self.assertGreater(len(features), 0)
        except ImportError:
            self.skipTest("无法导入extract_features函数")
    
    def test_image_processing(self):
        """测试图像处理功能"""
        # 创建一个渐变图像
        gradient = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient[:, i, 0] = 255 * (i / 100)  # 蓝色渐变
            gradient[:, i, 1] = 255 * (1 - i / 100)  # 绿色渐变
            gradient[:, i, 2] = 128  # 红色固定
        
        # 检查图像尺寸
        self.assertEqual(gradient.shape, (100, 100, 3))
        
        # 检查渐变效果
        self.assertGreater(gradient[50, 75, 0], gradient[50, 25, 0])  # 蓝色应该从左到右增加
        self.assertLess(gradient[50, 75, 1], gradient[50, 25, 1])  # 绿色应该从左到右减少
    
    def test_file_paths(self):
        """测试文件路径"""
        # 检查必要的目录是否存在
        self.assertTrue(os.path.exists("docs"), "docs目录不存在")
        self.assertTrue(os.path.exists("examples"), "examples目录不存在")
        
        # 检查必要的文件是否存在
        self.assertTrue(os.path.exists("README.md"), "README.md不存在")
        self.assertTrue(os.path.exists("requirements.txt"), "requirements.txt不存在")
        self.assertTrue(os.path.exists("setup.py"), "setup.py不存在")

if __name__ == "__main__":
    unittest.main()
