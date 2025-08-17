#!/usr/bin/env python3
"""
混色字符分类器训练脚本

此脚本用于训练一个分类器，用于识别图像中的混色字符（如蓝白渐变字符）。
通过提取字符图像的颜色分布特征，训练机器学习模型来区分混色字符和普通字符。
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# 目录配置
TRAINING_DATA_DIR = "training_data"  # 训练数据目录
TEST_DATA_DIR = "test_data"  # 测试数据目录
MIXED_COLOR_DIR = "mixed_color"  # 混色字符目录
NORMAL_COLOR_DIR = "normal_color"  # 普通字符目录
MODEL_DIR = "models"  # 模型保存目录
MODEL_FILE = os.path.join(MODEL_DIR, "mixed_color_classifier.joblib")  # 模型文件
RESULTS_DIR = "mixed_color_results"  # 结果保存目录

# 标注类别
MIXED_COLOR = 1  # 混色
NORMAL = 0       # 普通

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_features(image):
    """
    从字符图像中提取特征
    
    参数:
    image: 输入图像，OpenCV格式（BGR）
    
    返回:
    特征向量，包含颜色分布、梯度和纹理特征
    """
    # 确保图像不为空
    if image is None or image.size == 0:
        return np.zeros(10)  # 返回空特征
    
    # 调整图像大小为标准尺寸
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
    
    # 特征25-26: 蓝色倾向（B通道值高于R和G）
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
def load_data():
    """
    加载训练和测试数据
    
    返回:
    train_data, test_data: 训练数据和测试数据列表
    """
    # 训练数据目录
    train_mixed_dir = os.path.join(TRAINING_DATA_DIR, MIXED_COLOR_DIR)
    train_normal_dir = os.path.join(TRAINING_DATA_DIR, NORMAL_COLOR_DIR)
    
    # 测试数据目录
    test_mixed_dir = os.path.join(TEST_DATA_DIR, MIXED_COLOR_DIR)
    test_normal_dir = os.path.join(TEST_DATA_DIR, NORMAL_COLOR_DIR)
    
    # 检查目录是否存在
    for directory in [train_mixed_dir, train_normal_dir, test_mixed_dir, test_normal_dir]:
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return None, None
    
    # 加载训练数据
    train_data = []
    
    # 加载混色字符
    for filename in os.listdir(train_mixed_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            train_data.append({
                'image_path': os.path.join(train_mixed_dir, filename),
                'label': MIXED_COLOR
            })
    
    # 加载普通字符
    for filename in os.listdir(train_normal_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            train_data.append({
                'image_path': os.path.join(train_normal_dir, filename),
                'label': NORMAL
            })
    
    # 加载测试数据
    test_data = []
    
    # 加载混色字符
    for filename in os.listdir(test_mixed_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            test_data.append({
                'image_path': os.path.join(test_mixed_dir, filename),
                'label': MIXED_COLOR
            })
    
    # 加载普通字符
    for filename in os.listdir(test_normal_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            test_data.append({
                'image_path': os.path.join(test_normal_dir, filename),
                'label': NORMAL
            })
    
    print(f"加载了 {len(train_data)} 个训练样本")
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 统计训练数据类别
    train_mixed_count = sum(1 for item in train_data if item['label'] == MIXED_COLOR)
    train_normal_count = sum(1 for item in train_data if item['label'] == NORMAL)
    print(f"训练数据: 混色={train_mixed_count}, 普通={train_normal_count}")
    
    # 统计测试数据类别
    test_mixed_count = sum(1 for item in test_data if item['label'] == MIXED_COLOR)
    test_normal_count = sum(1 for item in test_data if item['label'] == NORMAL)
    print(f"测试数据: 混色={test_mixed_count}, 普通={test_normal_count}")
    
    return train_data, test_data

def prepare_dataset(data):
    """
    准备数据集
    
    参数:
    data: 数据列表，每个元素为包含image_path和label的字典
    
    返回:
    X: 特征矩阵
    y: 标签向量
    """
    X = []  # 特征
    y = []  # 标签
    
    for item in tqdm(data, desc="准备数据"):
        # 读取图像
        image = cv2.imread(item['image_path'])
        
        # 提取特征
        features = extract_features(image)
        
        # 添加到数据集
        X.append(features)
        y.append(item['label'])
    
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_test, y_test):
    """
    训练和评估模型
    
    参数:
    X_train, y_train: 训练数据
    X_test, y_test: 测试数据
    
    返回:
    best_model: 训练好的最佳模型
    y_pred: 预测结果
    """
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 创建模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])
    }
    
    # 训练和评估模型
    best_model = None
    best_score = 0
    y_pred = None
    
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"交叉验证分数: {cv_scores.mean():.4f} ±{cv_scores.std():.4f}")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 在测试集上评估
        score = model.score(X_test, y_test)
        print(f"测试集分数: {score:.4f}")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['普通', '混色']))
    
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
        
        # 更新最佳模型
        if score > best_score:
            best_score = score
            best_model = model
    
    return best_model, y_pred

def save_model(model):
    """保存模型"""
    ensure_dir(MODEL_DIR)
    joblib.dump(model, MODEL_FILE)
    print(f"模型已保存到 {MODEL_FILE}")

def visualize_results(model, X_test, y_test, y_pred, test_data):
    """可视化结果"""
    # 找出错误分类的样本
    errors = np.where(y_pred != y_test)[0]
    
    # 如果有错误分类的样本，可视化它们
    if len(errors) > 0:
        print(f"\n找到 {len(errors)} 个错误分类的样本")
        
        # 创建可视化目录
        vis_dir = os.path.join(RESULTS_DIR, "visualization")
        ensure_dir(vis_dir)
        
        # 可视化错误样本
        for i, idx in enumerate(errors[:10]):  # 只显示前10个错误
            # 获取原始图像
            image_path = test_data[idx]['image_path']
            image = cv2.imread(image_path)
            
            # 获取真实标签和预测标签
            true_label = "混色" if y_test[idx] == MIXED_COLOR else "普通"
            pred_label = "混色" if y_pred[idx] == MIXED_COLOR else "普通"
            
            # 创建可视化图像
            vis_image = image.copy()
            
            # 添加标签
            cv2.putText(vis_image, f"真实: {true_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_image, f"预测: {pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 保存可视化图像
            output_path = os.path.join(vis_dir, f"error_{i+1}.png")
            cv2.imwrite(output_path, vis_image)
            
            print(f"错误样本 {i+1}: 真实={true_label}, 预测={pred_label}, 保存到 {output_path}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['普通', '混色'], rotation=45)
    plt.yticks(tick_marks, ['普通', '混色'])
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存混淆矩阵图
    ensure_dir(RESULTS_DIR)
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"混淆矩阵已保存到 {cm_path}")
    
    # 保存错误分类的字符列表
    error_chars_file = os.path.join(RESULTS_DIR, "mixed_color_chars.txt")
    with open(error_chars_file, 'w', encoding='utf-8') as f:
        f.write("错误分类的字符:\n")
        for i, idx in enumerate(errors):
            image_path = test_data[idx]['image_path']
            true_label = "混色" if y_test[idx] == MIXED_COLOR else "普通"
            pred_label = "混色" if y_pred[idx] == MIXED_COLOR else "普通"
            f.write(f"{i+1}. {os.path.basename(image_path)}: 真实={true_label}, 预测={pred_label}\n")
    
    print(f"错误分类的字符列表已保存到 {error_chars_file}")

def main():
    """主函数"""
    # 加载数据
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # 准备训练数据集
    print("\n准备训练数据...")
    X_train, y_train = prepare_dataset(train_data)
    
    # 准备测试数据集
    print("\n准备测试数据...")
    X_test, y_test = prepare_dataset(test_data)
    
    # 训练模型
    print("\n训练模型...")
    model, y_pred = train_model(X_train, y_train, X_test, y_test)
    
    # 保存模型
    save_model(model)
    
    # 可视化结果
    visualize_results(model, X_test, y_test, y_pred, test_data)

if __name__ == "__main__":
    main()
