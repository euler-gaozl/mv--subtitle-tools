#!/usr/bin/env python3
"""
混色字符检测示例

此示例演示如何使用MV字幕工具集检测图像中的混色字符。
"""

import os
import sys
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from train_mixed_color_classifier import extract_features
import joblib

def load_model(model_path):
    """
    加载混色字符分类模型
    
    参数:
    model_path: 模型文件路径
    
    返回:
    加载的模型
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行train_mixed_color_classifier.py训练模型")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def detect_mixed_color(image_path, model, output_dir=None):
    """
    检测图像中的混色字符
    
    参数:
    image_path: 输入图像路径
    model: 混色字符分类模型
    output_dir: 输出目录，如果提供则保存可视化结果
    
    返回:
    is_mixed_color: 是否包含混色字符
    confidence: 置信度
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像: {image_path}")
        return False, 0.0
    
    # 提取特征
    features = extract_features(image)
    
    # 预测
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([features])[0]
        prediction = model.predict([features])[0]
        confidence = proba[prediction]
    else:
        prediction = model.predict([features])[0]
        confidence = 0.5  # 如果模型不支持概率输出，使用默认值
    
    is_mixed_color = prediction == 1  # 1表示混色
    
    # 可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建可视化图像
        vis_image = image.copy()
        
        # 添加预测结果
        label = "混色字符" if is_mixed_color else "普通字符"
        cv2.putText(vis_image, f"{label} ({confidence:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"result_{Path(image_path).stem}.png")
        cv2.imwrite(output_path, vis_image)
        print(f"结果已保存到: {output_path}")
        
        # 可视化特征
        visualize_features(features, image, os.path.join(output_dir, f"features_{Path(image_path).stem}.png"))
    
    return is_mixed_color, confidence

def visualize_features(features, image, output_path):
    """
    可视化特征
    
    参数:
    features: 特征向量
    image: 原始图像
    output_path: 输出路径
    """
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("原始图像")
    axs[0, 0].axis("off")
    
    # 左右BGR值对比
    left_bgr = features[0:3]
    right_bgr = features[3:6]
    
    bar_width = 0.35
    x = np.arange(3)
    
    axs[0, 1].bar(x - bar_width/2, left_bgr, bar_width, label="左侧")
    axs[0, 1].bar(x + bar_width/2, right_bgr, bar_width, label="右侧")
    axs[0, 1].set_title("左右BGR值对比")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(["B", "G", "R"])
    axs[0, 1].legend()
    
    # 蓝色和白色比例
    blue_ratios = features[16:19]  # [left_blue_ratio, right_blue_ratio, blue_ratio_diff]
    white_ratios = features[19:22]  # [left_white_ratio, right_white_ratio, white_ratio_diff]
    
    x = np.arange(2)
    
    axs[1, 0].bar(x - bar_width/2, [blue_ratios[0], white_ratios[0]], bar_width, label="左侧")
    axs[1, 0].bar(x + bar_width/2, [blue_ratios[1], white_ratios[1]], bar_width, label="右侧")
    axs[1, 0].set_title("蓝色和白色比例")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(["蓝色", "白色"])
    axs[1, 0].legend()
    
    # 特征重要性
    feature_names = [
        "左B", "左G", "左R", "右B", "右G", "右R", "亮度差", "颜色差",
        "左H", "左S", "左V", "右H", "右S", "右V", "HSV差",
        "左蓝比例", "右蓝比例", "蓝比例差", "左白比例", "右白比例", "白比例差",
        "左蓝倾向", "右蓝倾向"
    ]
    
    # 选择一些重要特征显示
    important_indices = [0, 3, 6, 7, 16, 17, 18, 19, 20, 21]
    important_features = [features[i] for i in important_indices]
    important_names = [feature_names[i] for i in important_indices]
    
    axs[1, 1].bar(range(len(important_features)), important_features)
    axs[1, 1].set_title("重要特征值")
    axs[1, 1].set_xticks(range(len(important_features)))
    axs[1, 1].set_xticklabels(important_names, rotation=45)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"特征可视化已保存到: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混色字符检测示例")
    parser.add_argument("image_path", help="输入图像路径")
    parser.add_argument("--model", default="../models/mixed_color_classifier.joblib", help="模型文件路径")
    parser.add_argument("--output", default="output", help="输出目录")
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model)
    if model is None:
        return
    
    # 检测混色字符
    is_mixed_color, confidence = detect_mixed_color(args.image_path, model, args.output)
    
    # 输出结果
    result = "混色字符" if is_mixed_color else "普通字符"
    print(f"检测结果: {result} (置信度: {confidence:.2f})")

if __name__ == "__main__":
    main()
