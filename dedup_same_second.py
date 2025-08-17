#!/usr/bin/env python3
"""
对现有的混色字符结果进行同一秒内去重
"""

import os
import json
import re

# 配置
RESULTS_DIR = "processing_results"
JSON_FILE = os.path.join(RESULTS_DIR, "all_results.json")
TXT_FILE = os.path.join(RESULTS_DIR, "mixed_characters.txt")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "all_results_deduped.json")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "mixed_characters_deduped.txt")

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

def process_results():
    """处理现有结果文件"""
    print("读取JSON结果文件...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {JSON_FILE}")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {JSON_FILE}")
        return
    
    print("对每个文件夹的结果进行同一秒内去重...")
    deduped_results = {}
    
    for folder, data in results.items():
        print(f"处理文件夹: {folder}")
        original = data.get("original", [])
        
        # 对原始数据进行去重
        deduped = deduplicate_same_second(original)
        
        # 添加到结果字典
        deduped_results[folder] = {
            "original": original,
            "deduplicated": deduped
        }
        
        # 打印统计信息
        print(f"  原始: {len(original)} 个字符")
        print(f"  去重后: {len(deduped)} 个字符")
        print(f"  减少了: {len(original) - len(deduped)} 个字符")
    
    # 保存JSON结果
    print(f"保存去重后的JSON结果到 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(deduped_results, f, ensure_ascii=False, indent=2)
    
    # 保存TXT结果
    print(f"保存去重后的TXT结果到 {OUTPUT_TXT}...")
    save_characters_to_file(deduped_results, OUTPUT_TXT)
    
    print("处理完成!")

if __name__ == "__main__":
    process_results()
