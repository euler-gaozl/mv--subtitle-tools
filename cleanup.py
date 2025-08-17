#!/usr/bin/env python3
"""
清理不需要的文件和文件夹
"""

import os
import shutil
import sys

# 要删除的文件
files_to_delete = [
    "annotation_tool.py",
    "batch_detect_mixed_color.py",
    "detect_mixed_color_improved.py",
    "detect_mixed_color.py",
    "extract_characters.py",
    "test_ocr_direct.py",
    "test_ocr_output.py",
    "test_ocr.py",
    "test.py",
    "train_classifier.py",
    "blue.png",
    "white.png",
    "mix.png"
]

# 要删除的文件夹
dirs_to_delete = [
    "detection_results",
    "mixed_color_results",
    "ocr_test_output",
    "test_data",
    "training_data"
]

def cleanup():
    """删除不需要的文件和文件夹"""
    print("=== 开始清理 ===")
    
    # 删除文件
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"已删除文件: {file}")
            except Exception as e:
                print(f"删除文件 {file} 时出错: {e}")
        else:
            print(f"文件不存在: {file}")
    
    # 删除文件夹
    for directory in dirs_to_delete:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"已删除文件夹: {directory}")
            except Exception as e:
                print(f"删除文件夹 {directory} 时出错: {e}")
        else:
            print(f"文件夹不存在: {directory}")
    
    print("\n=== 清理完成 ===")
    print("保留的核心文件:")
    print("- process_mixed_color_batch.py - 主要处理脚本")
    print("- train_mixed_color_classifier.py - 训练脚本")
    print("- ocr_utils.py - OCR工具")
    print("- models/ - 训练好的模型")
    print("- processing_output/ - 待处理的图片")
    print("- test/ - 测试图片")
    print("- processing_results/ - 处理结果")

if __name__ == "__main__":
    # 确认删除
    print("此脚本将删除以下文件和文件夹:")
    print("\n文件:")
    for file in files_to_delete:
        print(f"- {file}")
    
    print("\n文件夹:")
    for directory in dirs_to_delete:
        print(f"- {directory}/")
    
    confirm = input("\n确定要删除这些文件和文件夹吗? (y/n): ")
    
    if confirm.lower() == 'y':
        cleanup()
    else:
        print("操作已取消")
