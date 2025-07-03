#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import librosa
import soundfile
import numpy as np
import re
import configparser
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

def load_model_config(config_path: str = "model.conf") -> Dict[str, str]:
    """加载模型配置文件"""
    model_paths = {}
    
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 不存在，将使用在线模型")
        return model_paths
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        # 读取各类模型路径
        sections_map = {
            'asr_models_dir': 'asr',
            'vad_models_dir': 'vad', 
            'punc_models_dir': 'punc',
            'spk_models_dir': 'spk'
        }
        
        for section, model_type in sections_map.items():
            if section in config:
                for key, value in config[section].items():
                    model_paths[f"{model_type}_{key}"] = value
                    
        print(f"成功加载模型配置: {len(model_paths)} 个模型路径")
        for key, path in model_paths.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        print(f"读取配置文件失败: {e}")
    
    return model_paths

def convert_wav_ffmpeg(source_file: str, target_file: str, sample_rate: int):
    """使用ffmpeg转换音频文件"""
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    cmd = ["ffmpeg", "-y", "-i", source_file, "-ar", f"{sample_rate}", "-ac", "1", "-v", "quiet", target_file]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败 {source_file}: {e}")
        return False

def convert_files(source_dir: str, target_dir: str, sample_rate: int):
    """批量转换音频文件"""
    print(f"开始转换音频文件: {source_dir} -> {target_dir}")
    
    converted_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                continue
                
            source_path = os.path.join(root, file)
            rel_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, os.path.splitext(rel_path)[0] + '.wav')
            
            if not os.path.exists(target_path):
                if convert_wav_ffmpeg(source_path, target_path, sample_rate):
                    converted_files.append(target_path)
            else:
                converted_files.append(target_path)
    
    return converted_files

def ends_with_ending_sentence(sentence: str) -> bool:
    """检查句子是否以结束标点符号结尾"""
    return bool(re.search(r'[。？！…]$', sentence))

def ends_with_punctuation(sentence: str) -> bool:
    """检查句子是否以标点符号结尾"""
    pattern = r'[.,!?。，！？、・\uff00-\uffef\u3000-\u303f\u3040-\u309f\u30a0-\u30ff]$'
    return bool(re.search(pattern, sentence))

def merge_audio_slice(source_audio: str, slice_dir: str, data_list: List[Dict], 
                     start_count: int, sample_rate: int, max_seconds: int, 
                     language: str, audio_name: str) -> tuple:
    """合并音频切片并生成训练数据"""
    sentence_list = []
    audio_list = []
    time_length = 0
    count = start_count
    result = []

    data, sample_rate = librosa.load(source_audio, sr=sample_rate, mono=True)
    
    for sentence in data_list:
        text = sentence['text'].strip()
        if text == "":
            continue
            
        start = int(sentence['start'] * sample_rate)
        end = int(sentence['end'] * sample_rate)

        # 如果当前累积时长超过最大限制，保存当前片段
        if time_length > 0 and time_length + (sentence['end'] - sentence['start']) > max_seconds:
            sliced_audio_name = f"{audio_name}_{str(count).zfill(6)}"
            sliced_audio_path = os.path.join(slice_dir, sliced_audio_name + ".wav")
            s_sentence = "".join(sentence_list)

            # 中文标点符号处理
            if language == "ZH" and re.search(r"[，]$", s_sentence):
                s_sentence = s_sentence[:-1] + '。'

            audio_concat = np.concatenate(audio_list)
            if time_length > max_seconds:
                print(f"[音频过长]: {sliced_audio_path}, 时长: {time_length:.2f}秒")
            
            soundfile.write(sliced_audio_path, audio_concat, sample_rate)
            result.append({
                'sliced_audio_path': sliced_audio_path,
                'language': language,
                'text': s_sentence
            })
            
            sentence_list = []
            audio_list = []
            time_length = 0
            count += 1

        sentence_list.append(text)
        audio_list.append(data[start:end])
        time_length += (sentence['end'] - sentence['start'])
        
        # 如果遇到句子结束标点，分割保存
        if ends_with_ending_sentence(text):
            sliced_audio_name = f"{audio_name}_{str(count).zfill(6)}"
            sliced_audio_path = os.path.join(slice_dir, sliced_audio_name + ".wav")
            s_sentence = "".join(sentence_list)
            audio_concat = np.concatenate(audio_list)
            soundfile.write(sliced_audio_path, audio_concat, sample_rate)
            
            result.append({
                'sliced_audio_path': sliced_audio_path,
                'language': language,
                'text': s_sentence
            })
            
            sentence_list = []
            audio_list = []
            time_length = 0
            count += 1
    
    # 处理剩余的音频片段
    if sentence_list and audio_list:
        sliced_audio_name = f"{audio_name}_{str(count).zfill(6)}"
        sliced_audio_path = os.path.join(slice_dir, sliced_audio_name + ".wav")
        s_sentence = "".join(sentence_list)
        
        if language == "ZH" and re.search(r"[，]$", s_sentence):
            s_sentence = s_sentence[:-1] + '。'
            
        audio_concat = np.concatenate(audio_list)
        soundfile.write(sliced_audio_path, audio_concat, sample_rate)
        
        result.append({
            'sliced_audio_path': sliced_audio_path,
            'language': language,
            'text': s_sentence
        })
        count += 1

    return result, count

class FunASRProcessor:
    """FunASR处理器 - 支持本地模型配置"""
    
    def __init__(self, language: str = "ZH", model_type: str = "paraformer", 
                 model_paths: Optional[Dict[str, str]] = None):
        self.language = language
        self.model_type = model_type
        self.model_paths = model_paths or {}
        self.model = None
        self.init_model()
    
    def get_model_path(self, model_key: str, fallback_name: str) -> Optional[str]:
        """获取模型路径，优先使用本地配置"""
        if model_key in self.model_paths:
            local_path = self.model_paths[model_key]
            if os.path.exists(local_path):
                print(f"使用本地模型: {model_key} -> {local_path}")
                return local_path
            else:
                print(f"警告: 本地模型路径不存在: {local_path}")
        
        print(f"使用在线模型: {fallback_name}")
        return fallback_name
    
    def init_model(self):
        """初始化FunASR模型"""
        try:
            from funasr import AutoModel
            
            if self.language == "ZH":
                if self.model_type == "paraformer":
                    # 中文Paraformer模型 - 带VAD和标点
                    
                    # 获取模型路径
                    asr_model = self.get_model_path(
                        "asr_speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        "paraformer-zh"
                    )
                    vad_model = self.get_model_path(
                        "vad_speech_fsmn_vad_zh-cn-16k-common-pytorch", 
                        "fsmn-vad"
                    )
                    punc_model = self.get_model_path(
                        "punc_punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                        "ct-punc-c"
                    )
                    
                    # 构建模型参数
                    model_kwargs = {
                        "model": asr_model,
                        "vad_model": vad_model,
                        "punc_model": punc_model,
                    }
                    
                    # 只有在线模型需要指定版本
                    if asr_model == "paraformer-zh":
                        model_kwargs["model_revision"] = "v2.0.4"
                    if vad_model == "fsmn-vad":
                        model_kwargs["vad_model_revision"] = "v2.0.4"
                    if punc_model == "ct-punc-c":
                        model_kwargs["punc_model_revision"] = "v2.0.4"
                    
                    self.model = AutoModel(**model_kwargs)
                    
                else:
                    # 使用SenseVoice多语言模型
                    sensevoice_model = self.get_model_path(
                        "asr_sensevoice-small",
                        "sensevoice-small"
                    )
                    self.model = AutoModel(model=sensevoice_model)
            else:
                # 其他语言使用SenseVoice
                sensevoice_model = self.get_model_path(
                    "asr_sensevoice-small",
                    "sensevoice-small"
                )
                self.model = AutoModel(model=sensevoice_model)
                
            print(f"FunASR模型初始化成功 (语言: {self.language})")
            
        except ImportError:
            print("错误: 未找到funasr库")
            print("请安装: pip install funasr")
            raise
        except Exception as e:
            print(f"FunASR模型初始化失败: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> List[Dict]:
        """转录音频文件"""
        try:
            # 使用FunASR进行转录
            result = self.model.generate(input=audio_path)
            
            # 解析结果
            segments = []
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
                
                # 处理不同模型的输出格式
                if 'sentence_info' in res:
                    # Paraformer模型输出格式
                    for sentence in res['sentence_info']:
                        segments.append({
                            'start': sentence['start'] / 1000.0,  # 转换为秒
                            'end': sentence['end'] / 1000.0,
                            'text': sentence['text']
                        })
                elif 'text' in res:
                    # SenseVoice等模型输出格式
                    text = res['text']
                    if text.strip():
                        # 简单时间戳分配（如果没有详细时间戳）
                        segments.append({
                            'start': 0.0,
                            'end': self._get_audio_duration(audio_path),
                            'text': text
                        })
                else:
                    # 兜底处理
                    text_content = str(res) if res else ""
                    if text_content.strip():
                        segments.append({
                            'start': 0.0,
                            'end': self._get_audio_duration(audio_path),
                            'text': text_content
                        })
            
            return segments
            
        except Exception as e:
            print(f"转录失败 {audio_path}: {e}")
            return []
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except:
            return 10.0  # 默认10秒
    
    def __call__(self, audio_in: str) -> List[Dict]:
        """兼容原始接口"""
        return self.transcribe(audio_in)

def init_asr_model(language: str, max_seconds: int, model_paths: Optional[Dict[str, str]] = None):
    """初始化ASR模型"""
    return FunASRProcessor(language=language, model_paths=model_paths)

def create_dataset(converted_files: List[str], target_dir: str, sample_rate: int, 
                  language: str, infer_model, max_seconds: int, absolute_path: bool) -> List[str]:
    """创建数据集"""
    count = 0
    result = []

    print(f"发现 {len(converted_files)} 个音频文件")
    
    # 创建输出目录
    os.makedirs(target_dir, exist_ok=True)
    
    for audio_path in tqdm(converted_files, desc="处理音频文件"):
        try:
            print(f"正在识别: {audio_path}")
            data_list = infer_model(audio_in=audio_path)
            
            if not data_list:
                print(f"警告: {audio_path} 识别结果为空")
                continue
            
            # 使用音频文件名作为标识
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            
            data, count = merge_audio_slice(
                audio_path, target_dir, data_list, count, 
                sample_rate, max_seconds, language, audio_name
            )

            for item_audio in data:
                if absolute_path:
                    sliced_audio_path = os.path.abspath(item_audio['sliced_audio_path'])
                else:
                    sliced_audio_path = item_audio['sliced_audio_path']
                
                language = item_audio['language']
                text = item_audio['text']
                # 移除说话人字段，直接使用音频文件名作为标识
                audio_id = os.path.splitext(os.path.basename(sliced_audio_path))[0]
                result.append(f"{sliced_audio_path}|{audio_id}|{language}|{text}")
                
        except Exception as e:
            print(f"处理音频文件 {audio_path} 时出错: {e}")
            continue

    return result

def clean_cache_directory(cache_dir: str):
    """清除缓存目录"""
    try:
        if os.path.exists(cache_dir):
            print(f"\n5. 清理缓存目录: {cache_dir}")
            shutil.rmtree(cache_dir)
            print(f"   缓存目录已清理完成")
        else:
            print(f"\n5. 缓存目录不存在，无需清理: {cache_dir}")
    except Exception as e:
        print(f"清理缓存目录时出错: {e}")

def create_list(source_dir: str, target_dir: str, cache_dir: str, sample_rate: int, 
               language: str, output_list: str, max_seconds: int, absolute_path: bool,
               config_path: str = "model.conf", clean_cache: bool = True):
    """创建训练列表文件"""
    
    # 重采样目录
    resample_dir = os.path.join(cache_dir, "funasr_cache", "origin", f"{sample_rate}")
    
    print("=" * 60)
    print("开始创建数据集 (独立FunASR版本)")
    print("=" * 60)
    
    try:
        # 步骤0: 加载模型配置
        print(f"\n0. 加载模型配置文件: {config_path}")
        model_paths = load_model_config(config_path)
        
        # 步骤1: 转换音频文件
        print(f"\n1. 转换音频文件到指定采样率...")
        print(f"   源目录: {source_dir}")
        print(f"   缓存目录: {resample_dir}")
        converted_files = convert_files(source_dir, resample_dir, sample_rate)
        
        if not converted_files:
            print("错误: 没有找到可处理的音频文件")
            return
        
        print(f"   转换完成，共 {len(converted_files)} 个文件")
        
        # 步骤2: 初始化FunASR模型
        print(f"\n2. 初始化FunASR模型 (语言: {language})...")
        asr_model = init_asr_model(language, max_seconds, model_paths)
        
        # 步骤3: 处理音频并生成数据集
        print(f"\n3. 处理音频文件并生成数据集...")
        print(f"   输出目录: {target_dir}")
        result = create_dataset(
            converted_files, target_dir, sample_rate, 
            language, asr_model, max_seconds, absolute_path
        )
        
        # 步骤4: 保存结果到list文件
        print(f"\n4. 保存结果到 {output_list}...")
        os.makedirs(os.path.dirname(output_list) if os.path.dirname(output_list) else '.', exist_ok=True)
        
        with open(output_list, "w", encoding="utf-8") as file:
            for line in result:
                try:
                    file.write(line.strip() + '\n')
                except UnicodeEncodeError:
                    print(f"编码错误，跳过: {line}")
        
        print("=" * 60)
        print(f"处理完成！")
        print(f"生成 {len(result)} 条训练数据")
        print(f"输出文件: {output_list}")
        print(f"数据集目录: {target_dir}")
        print("=" * 60)
        
    finally:
        # 步骤5: 清理缓存目录（即使出现异常也要执行）
        if clean_cache:
            clean_cache_directory(cache_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用FunASR创建语音训练数据集（完全独立版本）")
    
    # 必需参数
    parser.add_argument("source_dir", type=str, 
                       help="源音频目录路径（必需）")
    parser.add_argument("target_dir", type=str, 
                       help="目标数据集目录路径（必需）")
    parser.add_argument("output", type=str, 
                       help="输出列表文件路径（必需）")
    
    # 可选参数
    parser.add_argument("--cache_dir", type=str, default="cache", 
                       help="缓存目录路径，默认: cache")
    parser.add_argument("--sample_rate", type=int, default=16000, 
                       help="采样率，默认: 16000")
    parser.add_argument("--language", type=str, default="ZH", 
                       help="语言代码，支持: ZH|EN|JA|KO|DE|RU等，默认: ZH")
    parser.add_argument("--max_seconds", type=int, default=10, 
                       help="最大音频片段长度(秒)，默认: 10")
    parser.add_argument("--relative_path", action="store_true", 
                       help="使用相对路径，默认使用绝对路径")
    parser.add_argument("--config", type=str, default="model.conf",
                       help="模型配置文件路径，默认: model.conf")
    parser.add_argument("--keep_cache", action="store_true",
                       help="保留缓存文件，默认处理完成后自动清理")
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    source_dir = os.path.abspath(args.source_dir)
    target_dir = os.path.abspath(args.target_dir)
    output_file = os.path.abspath(args.output)
    cache_dir = os.path.abspath(args.cache_dir)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        print("请确保目录中包含音频文件（支持格式: wav, mp3, flac, m4a, aac）")
        return 1
    
    # 检查是否有音频文件
    audio_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                audio_files.append(os.path.join(root, file))
    
    if not audio_files:
        print(f"错误: 在 {source_dir} 中未找到音频文件")
        print("支持的音频格式: wav, mp3, flac, m4a, aac")
        return 1
    
    print(f"发现 {len(audio_files)} 个音频文件")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"输出文件: {output_file}")
    print(f"缓存目录: {cache_dir}")
    print(f"配置文件: {args.config}")
    
    try:
        create_list(
            source_dir, target_dir, cache_dir, 
            args.sample_rate, args.language, output_file, 
            args.max_seconds, not args.relative_path,  # 默认使用绝对路径
            args.config, not args.keep_cache  # 默认清理缓存，除非指定保留
        )
        return 0
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 