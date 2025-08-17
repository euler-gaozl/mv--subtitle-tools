"""
OCR工具模块
提供字幕检测和文字颜色分析功能
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR


class OCRProcessor:
    """OCR处理器类，封装PaddleOCR的功能"""
    
    def __init__(self):
        """初始化OCR处理器"""
        self._ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """初始化PaddleOCR"""
        try:
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            print("OCR初始化成功")
        except Exception as e:
            print(f"OCR初始化失败: {e}")
            raise e
    
    def detect_text(self, image):
        """
        检测图像中的文字
        
        参数:
        image: 输入图像 (numpy array)
        
        返回:
        OCR检测结果
        """
        if self._ocr is None:
            raise RuntimeError("OCR未正确初始化")
        
        try:
            result = self._ocr.predict(image)
            return result
        except Exception as e:
            print(f"OCR检测失败: {e}")
            return None
    
    def get_text_regions(self, image, confidence_threshold=0.5):
        """
        获取图像中文字区域的坐标和文本
        
        参数:
        image: 输入图像
        confidence_threshold: 置信度阈值
        
        返回:
        list: [(bbox, text, confidence), ...]
        """
        result = self.detect_text(image)
        text_regions = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    bbox = line[0]  # 四个角点坐标
                    text, confidence = line[1]
                    
                    if confidence > confidence_threshold and len(text.strip()) > 0:
                        text_regions.append((bbox, text, confidence))
        
        return text_regions


def has_subtitle_text(image, ocr_processor):
    """
    检测图像中是否包含字幕文字
    
    参数:
    image: 输入图像 (numpy array)
    ocr_processor: OCR处理器实例
    
    返回:
    bool: True表示有文字，False表示无文字
    """
    try:
        result = ocr_processor.detect_text(image)
        
        # 检查是否检测到文字
        if result:
            # 兼容PaddleOCR的标准返回格式 [[[bbox], [text, confidence]], ...]
            if isinstance(result, list) and len(result) > 0:
                # 检查第一层列表是否有内容
                if result[0] and len(result[0]) > 0:
                    return True
            # 兼容其他可能的返回格式
            elif hasattr(result, 'get') and result.get('rec_texts'):
                if len(result['rec_texts']) > 0:
                    return True
        
        return False
        
    except Exception as e:
        print(f"文字检测出错: {e}")
        return False


def is_pure_color_text(image, ocr_processor, color_std_threshold=15):
    """
    检测图像中的文字是否为纯色
    
    参数:
    image: 输入图像 (numpy array)
    ocr_processor: OCR处理器实例
    color_std_threshold: 颜色标准差阈值
    
    返回:
    bool: True表示文字颜色纯净，False表示文字颜色不纯
    """
    try:
        # 获取文字区域
        text_regions = ocr_processor.get_text_regions(image)
        
        if not text_regions:
            return True  # 没有文字时认为是纯色
        
        # 分析每个文字区域的颜色纯度
        for bbox, text, confidence in text_regions:
            # 计算文字区域的边界框
            bbox = np.array(bbox, dtype=np.int32)
            x_min = max(0, int(np.min(bbox[:, 0])))
            y_min = max(0, int(np.min(bbox[:, 1])))
            x_max = min(image.shape[1], int(np.max(bbox[:, 0])))
            y_max = min(image.shape[0], int(np.max(bbox[:, 1])))
            
            # 提取文字区域像素
            text_region = image[y_min:y_max, x_min:x_max]
            
            if text_region.size > 0:
                # 分析文字颜色纯度
                if not _is_region_pure_color(text_region, color_std_threshold):
                    return False
        
        return True
        
    except Exception as e:
        print(f"颜色纯度检测出错: {e}")
        return True  # 出错时默认认为是纯色


def _is_region_pure_color(text_region, color_std_threshold):
    """
    检测文字区域的颜色是否纯净
    
    参数:
    text_region: 文字区域图像
    color_std_threshold: 颜色标准差阈值
    
    返回:
    bool: True表示颜色纯净
    """
    try:
        # 转换为灰度图进行文字像素检测
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        
        # 使用阈值分割找到文字像素（假设文字比背景暗）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 找到文字像素（值为0的像素）
        text_pixels = text_region[binary == 0]
        
        if len(text_pixels) > 10:  # 确保有足够的文字像素
            # 计算RGB三个通道的标准差
            b_std = np.std(text_pixels[:, 0])
            g_std = np.std(text_pixels[:, 1])
            r_std = np.std(text_pixels[:, 2])
            
            # 平均标准差
            avg_std = (b_std + g_std + r_std) / 3
            
            # 如果标准差过大，说明颜色不纯
            return avg_std <= color_std_threshold
        
        return True
        
    except Exception as e:
        print(f"区域颜色分析出错: {e}")
        return True


def analyze_text_color_stats(image, ocr_processor):
    """
    分析图像中文字的颜色统计信息
    
    参数:
    image: 输入图像
    ocr_processor: OCR处理器实例
    
    返回:
    dict: 包含颜色统计信息的字典
    """
    try:
        text_regions = ocr_processor.get_text_regions(image)
        stats = {
            'text_count': len(text_regions),
            'color_stats': []
        }
        
        for bbox, text, confidence in text_regions:
            bbox = np.array(bbox, dtype=np.int32)
            x_min = max(0, int(np.min(bbox[:, 0])))
            y_min = max(0, int(np.min(bbox[:, 1])))
            x_max = min(image.shape[1], int(np.max(bbox[:, 0])))
            y_max = min(image.shape[0], int(np.max(bbox[:, 1])))
            
            text_region = image[y_min:y_max, x_min:x_max]
            
            if text_region.size > 0:
                gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text_pixels = text_region[binary == 0]
                
                if len(text_pixels) > 10:
                    b_std = np.std(text_pixels[:, 0])
                    g_std = np.std(text_pixels[:, 1])
                    r_std = np.std(text_pixels[:, 2])
                    avg_std = (b_std + g_std + r_std) / 3
                    
                    mean_color = np.mean(text_pixels, axis=0)
                    
                    stats['color_stats'].append({
                        'text': text,
                        'confidence': confidence,
                        'avg_std': avg_std,
                        'mean_color': mean_color.tolist(),
                        'pixel_count': len(text_pixels)
                    })
        
        return stats
        
    except Exception as e:
        print(f"颜色统计分析出错: {e}")
        return {'text_count': 0, 'color_stats': []}
