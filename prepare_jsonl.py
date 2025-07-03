#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import re
import soundfile as sf
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# 设置并行处理的核心数
NUM_CORES = multiprocessing.cpu_count()
# 设置更大的批处理大小
BATCH_SIZE = 1000

def get_audio_info(audio_path):
    """获取音频文件信息"""
    if os.path.exists(audio_path):
        try:
            waveform, sr = sf.read(audio_path)
            sample_num = len(waveform)
            # 计算音频长度，单位为0.1秒
            context_len = int(sample_num * 1000 / sr / 10)
            return {"source": audio_path, "source_len": context_len}
        except Exception as e:
            print(f"处理音频文件 {audio_path} 时出错: {e}")
    return None

def process_batch(lines):
    """处理一批数据行"""
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('|', 3)
        if len(parts) != 4:
            print(f"警告: 行格式不正确: {line}")
            continue
        
        audio_path, _, _, text = parts  # 忽略说话人ID和语言字段
        
        # 获取音频文件信息
        audio_info = get_audio_info(audio_path)
        if not audio_info:
            continue
        
        # 构建数据项，仅包含所需字段
        item = {
            "key": os.path.basename(audio_path).split('.')[0],
            "source": audio_info["source"],
            "source_len": audio_info["source_len"],
            "target": text.strip(),
            "target_len": len(text.strip())
        }
        results.append(item)
    
    return results

def process_file_parallel(file_path):
    """并行处理输入文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"\n处理文件 {file_path}")
    print(f"总计 {total_lines} 条记录")
    
    # 将数据分成批次
    batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]
    total_batches = len(batches)
    
    all_results = []
    with tqdm(total=total_lines, desc="处理进度", unit="条") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                batch_results = future.result()
                all_results.extend(batch_results)
                pbar.update(min(BATCH_SIZE, len(lines) - (i-1)*BATCH_SIZE))
                pbar.set_postfix({"批次": f"{i}/{total_batches}"})
    
    return all_results

def split_and_save_jsonl(data, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    """将数据分割为训练集、验证集和测试集，并保存为jsonl文件"""
    # 随机打乱数据
    random.shuffle(data)
    total = len(data)
    
    # 计算各部分数据的索引
    train_split = int(total * train_ratio)
    val_split = int(total * (train_ratio + val_ratio))
    
    # 分割数据
    train_data = data[:train_split]
    val_data = data[train_split:val_split]
    test_data = data[val_split:]
    
    # 保存各部分数据为jsonl文件
    def write_jsonl(items, output_file):
        with open(output_file, "w", encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    write_jsonl(train_data, train_file)
    write_jsonl(val_data, val_file)
    write_jsonl(test_data, test_file)
    
    return len(train_data), len(val_data), len(test_data)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="从单个文件生成用于语音识别训练的jsonl文件")
    parser.add_argument("--input_file", required=True, help="输入文件路径，格式: 音频路径|讲话人ID|语言|文本内容")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置输出文件路径
    train_jsonl = os.path.join(args.output_dir, "train.jsonl")
    val_jsonl = os.path.join(args.output_dir, "val.jsonl")
    test_jsonl = os.path.join(args.output_dir, "test.jsonl")
    
    print(f"\n使用 {NUM_CORES} 个CPU核心并行处理")
    
    # 处理输入文件
    print("\n1. 处理输入文件...")
    data = process_file_parallel(args.input_file)
    
    # 如果没有有效数据，退出
    if not data:
        print("没有找到有效数据，退出。")
        return 1
    
    # 生成jsonl文件
    print("\n2. 生成训练集、验证集和测试集...")
    train_count, val_count, test_count = split_and_save_jsonl(
        data,
        train_file=train_jsonl,
        val_file=val_jsonl,
        test_file=test_jsonl,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print(f"\n处理完成：")
    print(f"- 训练集：{train_count} 样本")
    print(f"- 验证集：{val_count} 样本")
    print(f"- 测试集：{test_count} 样本")
    print(f"- 总计：{train_count + val_count + test_count} 样本")
    
    return 0

if __name__ == "__main__":
    exit(main())