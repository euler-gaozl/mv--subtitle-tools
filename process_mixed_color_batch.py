#!/usr/bin/env python3
"""
批量处理图片中的混色字符并进行连续去重
"""

import os
import sys
import cv2
import numpy as np
import joblib
import argparse
import glob
import json
import re
from ocr_utils import OCRProcessor

# 配置
MODEL_DIR = "models"  # 模型保存目录
MODEL_FILE = os.path.join(MODEL_DIR, "mixed_color_classifier.joblib")  # 模型文件
OUTPUT_DIR = "processing_results"  # 输出目录

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model():
    """加载模型"""
    try:
        model = joblib.load(MODEL_FILE)
        print(f"模型已加载: {MODEL_FILE}")
        return model
    except FileNotFoundError:
        print(f"模型文件不存在: {MODEL_FILE}")
        print("请先运行 train_mixed_color_classifier.py 训练模型")
        sys.exit(1)

def detect_characters(image_path, ocr_processor):
    """
    检测图片中的字符

    参数:
    image_path: 图片路径
    ocr_processor: OCR处理器实例

    返回:
    字符信息列表，每个字符包含文本、边界框、置信度等信息
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return [], None

    # 使用OCR检测文字
    result = ocr_processor.detect_text(image)

    # 处理OCR结果
    characters = []

    # 检查OCR结果是否有效
    if not result or len(result) == 0:
        return [], image

    # 获取OCR结果
    ocr_result = result[0]

    # 尝试从OCR结果中提取文本和边界框
    try:
        # 检查是否为字典类型或类似字典的对象
        if isinstance(ocr_result, dict) or hasattr(ocr_result, 'keys'):
            # 获取文本、置信度和多边形边界框
            if 'rec_texts' in ocr_result and 'rec_scores' in ocr_result and 'rec_polys' in ocr_result:
                texts = ocr_result['rec_texts']
                scores = ocr_result['rec_scores']
                polys = ocr_result['rec_polys']
                
                # 处理每个检测到的文本区域
                for text, score, poly in zip(texts, scores, polys):
                    # 计算边界框
                    poly_np = np.array(poly, dtype=np.int32)
                    x_min = max(0, int(np.min(poly_np[:, 0])))
                    y_min = max(0, int(np.min(poly_np[:, 1])))
                    x_max = min(image.shape[1], int(np.max(poly_np[:, 0])))
                    y_max = min(image.shape[0], int(np.max(poly_np[:, 1])))
                    
                    # 如果文本长度大于1，拆分为单个字符
                    if len(text) > 1:
                        # 估计每个字符的宽度
                        char_width = (x_max - x_min) // len(text)
                        
                        for i, char in enumerate(text):
                            # 计算字符的边界框
                            char_x_min = x_min + i * char_width
                            char_x_max = char_x_min + char_width
                            
                            characters.append({
                                'text': char,
                                'bbox': (char_x_min, y_min, char_x_max, y_max),
                                'confidence': float(score)
                            })
                    else:
                        # 单个字符
                        characters.append({
                            'text': text,
                            'bbox': (x_min, y_min, x_max, y_max),
                            'confidence': float(score)
                        })
    except Exception as e:
        print(f"处理OCR结果时出错: {e}")

    return characters, image

def extract_character_image(image, char_info):
    """
    从图像中提取字符区域

    参数:
    image: 输入图像
    char_info: 字符信息

    返回:
    字符图像
    """
    # 获取字符边界框
    x_min, y_min, x_max, y_max = char_info['bbox']

    # 提取字符区域
    char_image = image[y_min:y_max, x_min:x_max].copy()

    # 确保字符图像不为空
    if char_image.size == 0:
        return None, None

    return char_image, (x_min, y_min, x_max, y_max)

def detect_mixed_color_characters(image_path, model, ocr_processor):
    """
    检测图片中的混色字符

    参数:
    image_path: 图片路径
    model: 训练好的模型
    ocr_processor: OCR处理器实例

    返回:
    混色字符列表，每个元素为[字符, 图片名]
    """
    # 获取图片名
    image_name = os.path.basename(image_path)
    
    # 检测字符
    characters, image = detect_characters(image_path, ocr_processor)

    if image is None:
        return []

    # 如果没有检测到字符，返回空列表
    if not characters:
        return []

    # 混色字符列表
    mixed_color_chars = []

    # 处理每个字符
    for char_info in characters:
        # 提取字符图像
        char_result = extract_character_image(image, char_info)
        if char_result is None:
            continue

        char_image, bbox = char_result

        # 提取特征
        features = extract_features(char_image)

        # 预测
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]

        # 如果是混色字符，添加到列表
        if prediction == 1:  # 混色
            mixed_color_chars.append([char_info['text'], image_name])

    return mixed_color_chars

def process_folder_images(folder_path, model, ocr_processor, verbose=True):
    """
    处理文件夹中的所有图片

    参数:
    folder_path: 文件夹路径
    model: 训练好的模型
    ocr_processor: OCR处理器实例
    verbose: 是否打印详细信息

    返回:
    混色字符列表，每个元素为[字符, 图片名]
    """
    # 获取所有图片文件
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    
    # 按文件名排序（假设文件名包含时间戳）
    image_paths.sort()
    
    if verbose:
        print(f"找到 {len(image_paths)} 张图片")
    
    # 处理每张图片
    all_mixed_chars = []
    
    for i, image_path in enumerate(image_paths):
        if verbose and (i + 1) % 10 == 0:
            print(f"处理进度: {i + 1}/{len(image_paths)}")
        
        # 检测混色字符
        mixed_chars = detect_mixed_color_characters(image_path, model, ocr_processor)
        
        # 添加到结果列表
        all_mixed_chars.extend(mixed_chars)
    
    if verbose:
        print(f"共检测到 {len(all_mixed_chars)} 个混色字符")
    
    return all_mixed_chars

def sort_by_timestamp(data):
    """
    按时间戳排序

    参数:
    data: 混色字符列表，每个元素为[字符, 图片名]

    返回:
    排序后的列表
    """
    def extract_timestamp(item):
        # 提取时间戳
        match = re.search(r'(\d+_\d+_\d+_\d+)', item[1])
        if match:
            return match.group(1)
        return item[1]
    
    return sorted(data, key=extract_timestamp)

def extract_features(image):
    """
    从字符图像中提取特征

    参数:
    image: 输入图像

    返回:
    特征向量
    """
    # 确保图像不为空
    if image is None or image.size == 0:
        return np.zeros(10)  # 返回空特征

    # 调整图像大小
    image = cv2.resize(image, (64, 64))

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 水平分割为左右两部分
    h, w = hsv.shape[:2]
    left_hsv = hsv[:, :w//2]
    right_hsv = hsv[:, w//2:]
    left_bgr = image[:, :w//2]
    right_bgr = image[:, w//2:]

    # 特征1-6: 左右两侧的平均BGR值
    left_avg_bgr = np.mean(left_bgr, axis=(0, 1))
    right_avg_bgr = np.mean(right_bgr, axis=(0, 1))

    # 特征7-9: 左右两侧的亮度差异
    left_brightness = np.mean(left_bgr)
    right_brightness = np.mean(right_bgr)
    brightness_diff = abs(left_brightness - right_brightness)

    # 特征10-12: 左右两侧的颜色差异
    color_diff = np.sum(np.abs(left_avg_bgr - right_avg_bgr))

    # 特征13-15: 左右两侧的HSV平均值
    left_avg_hsv = np.mean(left_hsv, axis=(0, 1))
    right_avg_hsv = np.mean(right_hsv, axis=(0, 1))

    # 特征16-18: 左右两侧的HSV差异
    hsv_diff = np.sum(np.abs(left_avg_hsv - right_avg_hsv))

    # 特征19-21: 蓝色比例
    blue_lower = np.array([90, 100, 100])
    blue_upper = np.array([140, 255, 220])
    left_blue_mask = cv2.inRange(left_hsv, blue_lower, blue_upper)
    right_blue_mask = cv2.inRange(right_hsv, blue_lower, blue_upper)
    left_blue_ratio = np.sum(left_blue_mask > 0) / (left_hsv.shape[0] * left_hsv.shape[1])
    right_blue_ratio = np.sum(right_blue_mask > 0) / (right_hsv.shape[0] * right_hsv.shape[1])
    blue_ratio_diff = abs(left_blue_ratio - right_blue_ratio)

    # 特征22-24: 白色比例
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 50, 255])
    left_white_mask = cv2.inRange(left_hsv, white_lower, white_upper)
    right_white_mask = cv2.inRange(right_hsv, white_lower, white_upper)
    left_white_ratio = np.sum(left_white_mask > 0) / (left_hsv.shape[0] * left_hsv.shape[1])
    right_white_ratio = np.sum(right_white_mask > 0) / (right_hsv.shape[0] * right_hsv.shape[1])
    white_ratio_diff = abs(left_white_ratio - right_white_ratio)

    # 特征25: 蓝色倾向（B通道值高于R和G）
    blue_tendency_left = 1 if left_avg_bgr[0] > left_avg_bgr[1] + 10 and left_avg_bgr[0] > left_avg_bgr[2] + 10 else 0
    blue_tendency_right = 1 if right_avg_bgr[0] > right_avg_bgr[1] + 10 and right_avg_bgr[0] > right_avg_bgr[2] + 10 else 0

    # 组合所有特征
    features = np.concatenate([
        left_avg_bgr,
        right_avg_bgr,
        [brightness_diff],
        [color_diff],
        left_avg_hsv,
        right_avg_hsv,
        [hsv_diff],
        [left_blue_ratio, right_blue_ratio, blue_ratio_diff],
        [left_white_ratio, right_white_ratio, white_ratio_diff],
        [blue_tendency_left, blue_tendency_right]
    ])

    return features

def extract_second(filename):
    """
    从文件名中提取秒级时间戳
    格式: HH_MM_SS_mmm.png -> HH_MM_SS
    """
    match = re.search(r'(\d+_\d+_\d+)_\d+', filename)
    if match:
        return match.group(1)
    return filename  # 如果无法提取，返回原文件名

def deduplicate_same_second(data):
    """
    对同一秒内的重复字符进行去重
    
    参数:
    data: 混色字符列表，每个元素为[字符, 图片名]
    
    返回:
    去重后的列表
    """
    if not data:
        return []
    
    # 按时间戳排序
    sorted_data = sorted(data, key=lambda x: x[1])
    
    # 去重
    result = []
    seen_chars_in_second = {}  # 记录每一秒内已经出现的字符
    
    for char, filename in sorted_data:
        second = extract_second(filename)
        
        # 如果这个字符在这一秒内还没出现过，则保留
        if second not in seen_chars_in_second or char not in seen_chars_in_second[second]:
            if second not in seen_chars_in_second:
                seen_chars_in_second[second] = set()
            seen_chars_in_second[second].add(char)
            result.append([char, filename])
    
    return result

def deduplicate_consecutive_only(data, keep_single=False):
    """
    连续去重函数
    只保留连续出现的相同字符的第一个，可选是否保留单独出现的字符
    
    注意：此函数已被deduplicate_same_second替代，保留此函数是为了兼容性
    
    参数:
    data: 混色字符列表，每个元素为[字符, 图片名]
    keep_single: 是否保留单独出现的字符

    返回:
    去重后的列表
    """
    if not data:
        return []
    
    # 第一步：分组连续的相同字符
    groups = []
    current_group = [data[0]]
    
    for i in range(1, len(data)):
        if data[i][0] == data[i-1][0]:  # 如果当前字符与前一个字符相同
            current_group.append(data[i])
        else:
            groups.append(current_group)
            current_group = [data[i]]
    
    # 添加最后一组
    if current_group:
        groups.append(current_group)
    
    # 第二步：根据策略处理每组字符
    result = []
    for group in groups:
        if len(group) > 1:  # 连续出现的字符组
            result.append(group[0])  # 每组只保留第一个
        elif keep_single:  # 单独出现的字符
            result.append(group[0])  # 如果需要保留单独出现的字符
    
    return result

def test_with_test_folder():
    """
    用test文件夹测试功能
    """
    print("=== 测试模式 ===")
    
    # 加载模型
    model = load_model()
    
    # 初始化OCR处理器
    print("初始化OCR处理器...")
    ocr_processor = OCRProcessor()
    
    # 处理test文件夹
    test_folder = "test"
    print(f"处理文件夹: {test_folder}")
    
    # 检测混色字符
    mixed_chars = process_folder_images(test_folder, model, ocr_processor)
    
    # 按时间戳排序
    sorted_chars = sort_by_timestamp(mixed_chars)
    
    # 去重 - 使用同一秒内去重
    deduped_chars = deduplicate_same_second(sorted_chars)
    
    # 去重 - 使用旧的连续去重（保留单独出现的字符）
    deduped_chars_with_single = deduplicate_consecutive_only(sorted_chars, keep_single=True)
    
    # 打印结果
    print("\n原始检测结果:")
    for char in sorted_chars:
        print(f"字符: {char[0]}, 图片: {char[1]}")
    
    print("\n去重后结果 (不保留单独字符):")
    for char in deduped_chars:
        print(f"字符: {char[0]}, 图片: {char[1]}")
    
    print("\n去重后结果 (保留单独字符):")
    for char in deduped_chars_with_single:
        print(f"字符: {char[0]}, 图片: {char[1]}")
    
    # 保存结果
    ensure_dir(OUTPUT_DIR)
    result = {
        "original": sorted_chars,
        "deduplicated": deduped_chars,
        "deduplicated_with_single": deduped_chars_with_single
    }
    
    # 保存JSON结果
    output_json = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 保存字符到文本文件
    output_txt = os.path.join(OUTPUT_DIR, "test_characters.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=== 原始检测结果 ===\n")
        for char in sorted_chars:
            f.write(f"{char[0]}\t{char[1]}\n")
        
        f.write("\n=== 去重后结果 (不保留单独字符) ===\n")
        for char in deduped_chars:
            f.write(f"{char[0]}\t{char[1]}\n")
        
        f.write("\n=== 去重后结果 (保留单独字符) ===\n")
        for char in deduped_chars_with_single:
            f.write(f"{char[0]}\t{char[1]}\n")
    
    print(f"\n结果已保存到: {output_json}")
    print(f"字符列表已保存到: {output_txt}")
    
    # 测试去重逻辑
    test_deduplication()

def test_deduplication():
    """
    测试去重逻辑
    """
    print("\n=== 测试去重逻辑 ===")
    
    # 测试数据
    test_data = [
        ["啊", "00_00_11_11"],
        ["啊", "00_00_11_12"],
        ["啊", "00_00_11_13"],
        ["呢", "00_00_20_01"],  # 单独出现
        ["啊", "00_00_25_05"],  # 单独出现
        ["好", "00_00_30_10"],
        ["好", "00_00_30_11"],
        ["好", "00_00_30_12"],
        ["啊", "00_00_40_01"],  # 新一组连续
        ["啊", "00_00_40_02"]
    ]
    
    # 测试同一秒内去重
    print("\n测试同一秒内去重:")
    deduped_same_second = deduplicate_same_second(test_data)
    
    # 去重 - 旧的连续去重（不保留单独出现的字符）
    deduped = deduplicate_consecutive_only(test_data, keep_single=False)
    
    # 去重 - 旧的连续去重（保留单独出现的字符）
    deduped_with_single = deduplicate_consecutive_only(test_data, keep_single=True)
    
    # 打印结果
    print("原始数据:")
    for item in test_data:
        print(f"字符: {item[0]}, 图片: {item[1]}")
    
    print("\n同一秒内去重后:")
    for item in deduped_same_second:
        print(f"字符: {item[0]}, 图片: {item[1]}")
    
    print("\n旧算法去重后 (不保留单独字符):")
    for item in deduped:
        print(f"字符: {item[0]}, 图片: {item[1]}")
    
    print("\n旧算法去重后 (保留单独字符):")
    for item in deduped_with_single:
        print(f"字符: {item[0]}, 图片: {item[1]}")
    
    # 期望结果 - 同一秒内去重
    expected_same_second = [
        ["啊", "00_00_11_11"],  # 11秒内只保留第一个"啊"
        ["呢", "00_00_20_01"],  # 20秒内的"呢"
        ["啊", "00_00_25_05"],  # 25秒内的"啊"
        ["好", "00_00_30_10"],  # 30秒内只保留第一个"好"
        ["啊", "00_00_40_01"]   # 40秒内只保留第一个"啊"
    ]
    
    # 验证 - 同一秒内去重
    is_correct_same_second = len(deduped_same_second) == len(expected_same_second)
    if is_correct_same_second:
        for i in range(len(deduped_same_second)):
            if deduped_same_second[i][0] != expected_same_second[i][0]:
                is_correct_same_second = False
                break
    
    print(f"\n测试结果 (同一秒内去重): {'通过' if is_correct_same_second else '失败'}")

def save_characters_to_file(results, output_file):
    """
    将字符保存到文本文件

    参数:
    results: 结果字典，每个文件夹对应一个键
    output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for folder, data in results.items():
            f.write(f"=== {folder} ===\n")
            for char_info in data["deduplicated"]:
                char = char_info[0]
                image = char_info[1]
                f.write(f"{char}\t{image}\n")
            f.write("\n")

def process_all_folders(keep_single=False):
    """
    处理processing_output下的所有文件夹

    参数:
    keep_single: 是否保留单独出现的字符（此参数保留但不再使用）
    """
    print("=== 处理所有文件夹 ===")
    print("使用同一秒内去重算法")
    
    # 加载模型
    model = load_model()
    
    # 初始化OCR处理器
    print("初始化OCR处理器...")
    ocr_processor = OCRProcessor()
    
    # 获取processing_output下的所有文件夹
    base_dir = "processing_output"
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    # 结果字典
    results = {}
    
    # 处理每个文件夹
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        print(f"\n处理文件夹: {folder_path}")
        
        # 检测混色字符
        mixed_chars = process_folder_images(folder_path, model, ocr_processor)
        
        # 按时间戳排序
        sorted_chars = sort_by_timestamp(mixed_chars)
        
        # 去重 - 使用同一秒内去重
        deduped_chars = deduplicate_same_second(sorted_chars)
        
        # 添加到结果字典
        results[folder] = {
            "original": sorted_chars,
            "deduplicated": deduped_chars
        }
        
        # 打印结果
        print(f"原始检测结果: {len(sorted_chars)} 个混色字符")
        print(f"去重后结果: {len(deduped_chars)} 个混色字符")
    
    # 保存结果
    ensure_dir(OUTPUT_DIR)
    
    # 保存JSON结果
    output_json = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存字符到文本文件
    output_txt = os.path.join(OUTPUT_DIR, "mixed_characters.txt")
    save_characters_to_file(results, output_txt)
    
    print(f"\n所有结果已保存到: {output_json}")
    print(f"字符列表已保存到: {output_txt}")
    
    # 返回结果
    return results

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量处理混色字符检测')
    parser.add_argument('--test', action='store_true', help='测试模式，只处理test文件夹')
    parser.add_argument('--all', action='store_true', help='处理所有文件夹')
    parser.add_argument('--keep-single', action='store_true', 
                        help='保留单独出现的字符（已弃用，保留此参数是为了兼容性）')
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        test_with_test_folder()
    # 处理所有文件夹
    elif args.all:
        process_all_folders()
    # 默认测试模式
    else:
        test_with_test_folder()

if __name__ == "__main__":
    main()
