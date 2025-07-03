#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_text_to_files(list_file):
    """
    从list文件中提取每个音频对应的文本内容，保存为txt文件
    
    Args:
        list_file (str): 输入的list文件路径，格式: 音频路径|说话人ID|语言|文本内容
    """
    
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"输入文件不存在: {list_file}")
    
    success_count = 0
    error_count = 0
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"开始处理文件: {list_file}")
    print(f"总计 {len(lines)} 条记录")
    
    for line_num, line in enumerate(tqdm(lines, desc="提取文本"), 1):
        line = line.strip()
        if not line:
            continue
        
        # 解析每行数据
        parts = line.split('|', 3)
        if len(parts) != 4:
            print(f"警告: 第{line_num}行格式不正确: {line}")
            error_count += 1
            continue
        
        audio_path, speaker_id, language, text = parts
        text = text.strip()
        
        if not text:
            print(f"警告: 第{line_num}行文本内容为空")
            error_count += 1
            continue
        
        # 生成对应的txt文件路径
        audio_path_obj = Path(audio_path)
        txt_path = audio_path_obj.with_suffix('.txt')
        
        try:
            # 确保目录存在
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文本内容
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
            
            success_count += 1
            
        except Exception as e:
            print(f"错误: 处理第{line_num}行时出错: {e}")
            error_count += 1
    
    print(f"\n处理完成:")
    print(f"- 成功创建: {success_count} 个txt文件")
    print(f"- 处理失败: {error_count} 条记录")
    
    return success_count, error_count

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从list文件提取文本内容并保存为txt文件")
    parser.add_argument("input_file", help="输入的list文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要创建的文件，不实际创建")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("预览模式 - 仅显示将要创建的文件:")
        preview_extraction(args.input_file)
    else:
        extract_text_to_files(args.input_file)

def preview_extraction(list_file):
    """预览将要创建的txt文件"""
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"输入文件不存在: {list_file}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"预览文件: {list_file}")
    print(f"将要创建 {len([l for l in lines if l.strip()])} 个txt文件:\n")
    
    for line_num, line in enumerate(lines[:10], 1):  # 只显示前10条
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('|', 3)
        if len(parts) != 4:
            continue
        
        audio_path, _, _, text = parts
        txt_path = Path(audio_path).with_suffix('.txt')
        
        print(f"{line_num:3d}. {audio_path}")
        print(f"     -> {txt_path}")
        print(f"     文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        print()
    
    if len(lines) > 10:
        print(f"... 还有 {len(lines) - 10} 条记录")

if __name__ == "__main__":
    main()
