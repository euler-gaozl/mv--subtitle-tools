# 导入所需的库
import cv2
import os
import glob
import shutil
import numpy as np
from ocr_utils import OCRProcessor, has_subtitle_text, is_pure_color_text

# --- 全局配置 ---
# 1. 字幕区域 (ROI - Region of Interest)
#    这个坐标在我们后续的步骤中会用到
SUBTITLE_ROI = (27, 260, 656, 175)

# --- 步骤 1: 提取所有帧并以时间轴命名 ---
def extract_frames_with_timestamps(video_path, output_dir):
    """
    从视频中提取所有帧，并将它们以 '小时_分钟_秒_毫秒.png' 的格式保-存。

    参数:
    video_path (str): 视频文件的路径。
    output_dir (str): 保存所有提取帧的目录。
    """
    print(f"--- 步骤 1: 开始从 '{os.path.basename(video_path)}' 提取所有帧 ---")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return False

    # 清理或创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除旧的帧以避免混淆
    os.makedirs(output_dir)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的毫秒级时间戳
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # 将毫秒转换为时、分、秒、毫秒
        seconds = int(timestamp_ms / 1000)
        milliseconds = int(timestamp_ms % 1000)
        minutes = int(seconds / 60)
        hours = int(minutes / 60)
        
        # 格式化文件名
        # zfill(2) 用于补零，例如 5 -> 05
        filename = (f"{hours:02d}_{minutes % 60:02d}_{seconds % 60:02d}_{milliseconds:03d}.png")
        
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        frame_count += 1
        
        print(f"  > 已保存: {filename}", end='\r')

    cap.release()
    print(f"\n步骤 1 完成：共提取并保存了 {frame_count} 帧到 '{output_dir}'")
    return True

# --- 步骤 2: 删除不含字幕的帧 ---
def filter_frames_without_subtitles(frames_dir, subtitle_roi, ocr_processor=None):
    """
    遍历一个文件夹中的所有帧，删除那些在ROI区域内没有字幕的帧。
    使用OCR模块检测文字存在性。
    """
    print("\n--- 步骤 2: 开始过滤不含字幕的帧 ---")
    
    # 初始化OCR处理器（如果没有提供的话）
    if ocr_processor is None:
        try:
            ocr_processor = OCRProcessor()
        except Exception as e:
            print(f"OCR初始化失败: {e}")
            return
    
    # 获取所有图片文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(frames_dir, ext)))
    image_files = sorted(image_files)
    
    if not image_files:
        print("未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个帧文件，开始检测...")
    
    deleted_count = 0
    x, y, w, h = subtitle_roi
    
    for i, image_path in enumerate(image_files):
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # 裁剪字幕区域
            roi = img[y:y+h, x:x+w]
             
            # 使用OCR模块检测文字
            has_text = has_subtitle_text(roi, ocr_processor)
            
            # 如果没有检测到文字，删除该帧
            if not has_text:
                os.remove(image_path)
                deleted_count += 1
                print(f"  删除无字幕帧: {os.path.basename(image_path)}")
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"  已处理: {i + 1}/{len(image_files)}")
                
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
            continue
    
    print(f"步骤 2 完成：删除了 {deleted_count} 个无字幕帧，剩余 {len(image_files) - deleted_count} 帧")

# --- 步骤 3: 删除字幕为非纯色的帧 ---
def filter_frames_with_non_pure_color_text(frames_dir, subtitle_roi, ocr_processor=None, color_std_threshold=15):
    """
    遍历文件夹中剩下的帧，删除那些字幕文字颜色不纯的帧。
    使用OCR模块定位文字区域，然后分析颜色一致性。
    """
    print("\n--- 步骤 3: 开始过滤非纯色字幕的帧 ---")
    
    # 初始化OCR处理器（如果没有提供的话）
    if ocr_processor is None:
        try:
            ocr_processor = OCRProcessor()
        except Exception as e:
            print(f"OCR初始化失败: {e}")
            return
    
    # 获取所有图片文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(frames_dir, ext)))
    image_files = sorted(image_files)
    
    if not image_files:
        print("未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个帧文件，开始检测字体颜色...")
    
    deleted_count = 0
    x, y, w, h = subtitle_roi
    
    for i, image_path in enumerate(image_files):
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # 裁剪字幕区域
            roi = img[y:y+h, x:x+w]
            
            # 使用OCR模块检测文字颜色纯度
            pure_color = is_pure_color_text(roi, ocr_processor, color_std_threshold)
            
            # 如果检测到非纯色文字，删除该帧
            if not pure_color:
                os.remove(image_path)
                deleted_count += 1
                print(f"  删除非纯色字幕帧: {os.path.basename(image_path)}")
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"  已处理: {i + 1}/{len(image_files)}")
                
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
            continue
    
    print(f"步骤 3 完成：删除了 {deleted_count} 个非纯色字幕帧，最终剩余 {len(image_files) - deleted_count} 帧")

# --- 主程序执行入口 ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'video')
    # 我们将所有处理过程都放在一个新的 'processing_output' 文件夹中
    base_output_folder = os.path.join(script_dir, 'processing_output')

    if not os.path.isdir(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在。")
        exit()

    supported_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.ts']
    video_files = []
    for ext in supported_extensions:
        search_pattern = os.path.join(input_folder, '**', ext)
        video_files.extend(glob.glob(search_pattern, recursive=True))
    video_files = sorted(list(set(video_files)))

    if not video_files:
        print(f"错误：在 '{input_folder}' 文件夹中没有找到任何支持的视频文件。")
    else:
        print(f"找到了 {len(video_files)} 个视频文件。")
        
        # 初始化OCR处理器（只初始化一次，提高性能）
        try:
            ocr_processor = OCRProcessor()
            print("全局OCR处理器初始化成功")
        except Exception as e:
            print(f"全局OCR处理器初始化失败: {e}")
            print("将在每个处理步骤中单独初始化OCR")
            ocr_processor = None
        
        for video_path in video_files:
            print(f"\n{'='*20} 开始处理视频: {video_path} {'='*20}")
            
            # 为每个视频创建一个独立的子文件夹来存放其处理过程中的所有帧
            video_filename = os.path.splitext(os.path.basename(video_path))[0]
            processing_dir_for_video = os.path.join(base_output_folder, video_filename)
            
            # 执行处理管道
            # 步骤 1
            success = extract_frames_with_timestamps(video_path, processing_dir_for_video)
            
            if success:
                # 步骤 2 - 传入共享的OCR处理器
                filter_frames_without_subtitles(processing_dir_for_video, SUBTITLE_ROI, ocr_processor)
                
                # 步骤 3 - 传入共享的OCR处理器
                filter_frames_with_non_pure_color_text(processing_dir_for_video, SUBTITLE_ROI, ocr_processor)
            
            print(f"\n{'='*20} 视频 {video_path} 处理完毕 {'='*20}")

        print("\n--- 所有视频处理完毕 ---")
