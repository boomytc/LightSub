import os
import argparse
import subprocess
import librosa
import soundfile
import numpy as np
import re
import shutil
from tqdm import tqdm
from typing import List, Dict, Optional

"""
全局默认参数（可在此处直观修改）
说明：
- 位置参数 source_dir/target_dir/output 支持通过下方默认值省略命令行传参。
- 若保持为 None，则仍需通过命令行显式传入。
"""

# I/O 路径默认值（可改为你的实际路径）
DEFAULT_SOURCE_DIR: Optional[str] = None          # 例如："/path/to/raw_audios"
DEFAULT_TARGET_DIR: Optional[str] = "dataset/audio_split"          # 工作区根目录下的音频切片目录
DEFAULT_OUTPUT_FILE: Optional[str] = "dataset/audio_list/list.txt"         # 工作区根目录下的列表文件

# 处理流程默认参数
DEFAULT_CACHE_DIR: str = "cache"                  # 工作区根目录下的缓存目录
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_LANGUAGE: str = "ZH"                      # ZH/EN/JA/...
DEFAULT_MAX_SECONDS: int = 10
DEFAULT_USE_ABSOLUTE_PATH: bool = True            # True=绝对路径; False=相对路径（等价于命令行不/启用 --relative_path）
DEFAULT_KEEP_CACHE: bool = False

# 外部依赖/文件类型
FFMPEG_BIN: str = "ffmpeg"
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.aac')

# 本地模型目录（使用相对路径，便于直观修改）
ASR_PARAFORMER_DIR = "models/asr_models/paraformer"
ASR_SENSEVOICE_DIR = "models/asr_models/sensevoice"
PUNC_MODEL_DIR = "models/punc_models/punc_ct"
SPK_MODEL_DIR = "models/spk_models/campplus_sv"
VAD_MODEL_DIR = "models/vad_models/fsmn_vad"

def _ensure_exists(path: str, name: str) -> str:
    """检查本地模型目录是否存在，不存在则报错提示。"""
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"未找到{name}模型目录: {path}\n"
        f"请将对应模型放置在该目录，或修改脚本顶部路径配置。"
    )

def convert_wav_ffmpeg(source_file: str, target_file: str, sample_rate: int):
    """使用ffmpeg转换音频文件"""
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    
    cmd = [FFMPEG_BIN, "-y", "-i", source_file, "-ar", f"{sample_rate}", "-ac", "1", "-v", "quiet", target_file]
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
            if not file.lower().endswith(SUPPORTED_EXTENSIONS):
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
    """FunASR处理器 - 使用本地固定目录加载模型"""
    
    def __init__(self, language: str = "ZH"):
        self.language = language
        self.model = None
        self.init_model()
    
    def init_model(self):
        """初始化FunASR模型"""
        try:
            from funasr import AutoModel
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO
            
            if self.language == "ZH":
                # 中文使用 Paraformer 组合（ASR + VAD + PUNC + SPK）
                asr_model = _ensure_exists(ASR_PARAFORMER_DIR, "ASR(Paraformer)")
                vad_model = _ensure_exists(VAD_MODEL_DIR, "VAD(FSMN)")
                punc_model = _ensure_exists(PUNC_MODEL_DIR, "PUNC(CT)")
                spk_model = _ensure_exists(SPK_MODEL_DIR, "SPK(CampPlus)")

                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    self.model = AutoModel(
                        model=asr_model,
                        vad_model=vad_model,
                        vad_kwargs={"max_single_segment_time": 30000},
                        punc_model=punc_model,
                        spk_model=spk_model,
                        disable_update=True,
                    )
            else:
                # 非中文使用 SenseVoice + VAD 组合
                sensevoice_model = _ensure_exists(ASR_SENSEVOICE_DIR, "ASR(SenseVoice)")
                vad_model = _ensure_exists(VAD_MODEL_DIR, "VAD(FSMN)")
                
                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    self.model = AutoModel(
                        model=sensevoice_model,
                        vad_model=vad_model,
                        vad_kwargs={"max_single_segment_time": 30000},
                    )
                
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
            from contextlib import redirect_stderr, redirect_stdout
            from io import StringIO
            import re
            
            # 使用FunASR进行转录
            if self.language == "ZH":
                # 中文模型（Paraformer）
                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    result = self.model.generate(input=audio_path, cache={})
            else:
                # 非中文模型（SenseVoice）
                generate_kwargs = {
                    "language": "auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
                    "use_itn": True,
                    "batch_size_s": 60,
                    "merge_vad": True,  
                    "merge_length_s": 15,
                }
                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    result = self.model.generate(
                        input=audio_path,
                        cache={},
                        **generate_kwargs,
                        output_timestamp=True,
                    )
            
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
                elif 'text' in res and 'timestamp' in res:
                    # SenseVoice模型输出格式
                    text = res['text']
                    timestamps = res['timestamp']
                    
                    # 清理SenseVoice特殊标记
                    text = self._clean_sensevoice_text(text)
                    
                    if text.strip() and timestamps:
                        # 使用SenseVoice的时间戳信息
                        start_time = timestamps[0][0] / 1000.0 if timestamps else 0.0
                        end_time = timestamps[-1][1] / 1000.0 if timestamps else self._get_audio_duration(audio_path)
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })
                elif 'text' in res:
                    # 兜底：只有文本没有时间戳
                    text = res['text']
                    text = self._clean_sensevoice_text(text)
                    
                    if text.strip():
                        segments.append({
                            'start': 0.0,
                            'end': self._get_audio_duration(audio_path),
                            'text': text
                        })
                else:
                    # 最后兜底处理
                    text_content = str(res) if res else ""
                    text_content = self._clean_sensevoice_text(text_content)
                    
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
    
    def _clean_sensevoice_text(self, text: str) -> str:
        """清理SenseVoice输出中的特殊标记"""
        if not text:
            return ""
        
        import re
        # 移除SenseVoice特殊标记：<|zh|><|NEUTRAL|><|Speech|><|withitn|>等
        cleaned_text = re.sub(r'<\|[^|]*\|>', '', text)
        return cleaned_text.strip()
    
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

def init_asr_model(language: str):
    """初始化ASR模型（使用本地固定目录加载）"""
    return FunASRProcessor(language=language)

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
               clean_cache: bool = True):
    """创建训练列表文件"""
    
    # 重采样目录
    resample_dir = os.path.join(cache_dir, "funasr_cache", "origin", f"{sample_rate}")
    
    print("=" * 60)
    print("开始创建数据集 (独立FunASR版本)")
    print("=" * 60)
    
    try:
        # 步骤1: 转换音频文件
        print(f"\n1. 转换音频文件到指定采样率...")
        print(f"   源目录: {source_dir}")
        print(f"   缓存目录: {resample_dir}")
        converted_files = convert_files(source_dir, resample_dir, sample_rate)
        
        if not converted_files:
            print("错误: 没有找到可处理的音频文件")
            return
        
        print(f"   转换完成，共 {len(converted_files)} 个文件")
        
        # 步骤2: 初始化FunASR模型（本地固定目录）
        print(f"\n2. 初始化FunASR模型 (语言: {language})...")
        asr_model = init_asr_model(language)
        
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
    
    # 位置参数（可通过脚本顶部默认值省略）
    parser.add_argument("source_dir", nargs='?', default=DEFAULT_SOURCE_DIR, type=str,
                       help=f"源音频目录路径（可省略，默认见脚本顶部 DEFAULT_SOURCE_DIR）")
    parser.add_argument("target_dir", nargs='?', default=DEFAULT_TARGET_DIR, type=str,
                       help=f"目标数据集目录路径（可省略，默认见脚本顶部 DEFAULT_TARGET_DIR）")
    parser.add_argument("output", nargs='?', default=DEFAULT_OUTPUT_FILE, type=str,
                       help=f"输出列表文件路径（可省略，默认见脚本顶部 DEFAULT_OUTPUT_FILE）")
    
    # 可选参数
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR,
                       help=f"缓存目录路径，默认: {DEFAULT_CACHE_DIR}")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
                       help=f"采样率，默认: {DEFAULT_SAMPLE_RATE}")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE,
                       help=f"语言代码，支持: ZH|EN|JA|KO|DE|RU等，默认: {DEFAULT_LANGUAGE}")
    parser.add_argument("--max_seconds", type=int, default=DEFAULT_MAX_SECONDS,
                       help=f"最大音频片段长度(秒)，默认: {DEFAULT_MAX_SECONDS}")
    parser.add_argument("--relative_path", action="store_true",
                       help=f"使用相对路径，默认使用绝对路径（当前默认: {'绝对' if DEFAULT_USE_ABSOLUTE_PATH else '相对'}）")
    parser.add_argument("--keep_cache", action="store_true", default=DEFAULT_KEEP_CACHE,
                       help=f"保留缓存文件，默认处理完成后自动清理（当前默认: {'保留' if DEFAULT_KEEP_CACHE else '清理'}）")
    
    args = parser.parse_args()
    
    # 解析路径，允许顶部默认值生效
    def _abs_or_none(p: Optional[str]):
        return os.path.abspath(p) if p else None

    source_dir = _abs_or_none(args.source_dir)
    target_dir = _abs_or_none(args.target_dir) if args.target_dir else os.path.abspath(DEFAULT_TARGET_DIR)
    output_file = _abs_or_none(args.output) if args.output else os.path.abspath(DEFAULT_OUTPUT_FILE)
    cache_dir = _abs_or_none(args.cache_dir)

    if not source_dir:
        print("错误: 必须指定源音频目录 source_dir。")
        print()
        print("默认输出路径:")
        print(f"  目标目录: {DEFAULT_TARGET_DIR}")
        print(f"  列表文件: {DEFAULT_OUTPUT_FILE}")
        print(f"  缓存目录: {DEFAULT_CACHE_DIR}")
        print()
        print("使用方法:")
        print("  python datasets_create_list_standalone.py /path/to/source_audio")
        print("或在脚本顶部设置 DEFAULT_SOURCE_DIR 后直接运行。")
        return 1
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        print("请确保目录中包含音频文件（支持格式: wav, mp3, flac, m4a, aac）")
        return 1
    
    # 检查是否有音频文件
    audio_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
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
    # 打印本地模型目录（便于确认）
    print("本地模型目录:")
    print(f"  ASR(Paraformer): {ASR_PARAFORMER_DIR}")
    print(f"  ASR(SenseVoice): {ASR_SENSEVOICE_DIR}")
    print(f"  VAD(FSMN):       {VAD_MODEL_DIR}")
    print(f"  PUNC(CT):        {PUNC_MODEL_DIR}")
    print(f"  SPK(CampPlus):   {SPK_MODEL_DIR}")
    
    try:
        create_list(
            source_dir, target_dir, cache_dir, 
            args.sample_rate, args.language, output_file, 
            # 默认使用绝对路径；如需默认相对路径可在顶部改 DEFAULT_USE_ABSOLUTE_PATH
            args.max_seconds, DEFAULT_USE_ABSOLUTE_PATH and (not args.relative_path),
            not args.keep_cache  # 默认清理缓存，除非指定保留
        )
        return 0
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 
