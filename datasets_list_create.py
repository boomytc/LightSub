import os
import argparse
import subprocess
import shutil
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import librosa
from funasr import AutoModel
 

"""
使用FunASR创建语音训练数据集

数据格式说明：
输出格式：音频路径|文本内容
示例：/path/to/audio_000001.wav|这是识别出的文本内容

设计理念：
- 消除冗余的音频ID字段（可从路径实时提取：os.path.splitext(os.path.basename(path))[0]）
- 移除语言字段与语言代码指定（仅保留中英混合识别场景）
- 专注核心功能：音频-文本映射关系
- 简化处理流程，减少存储空间
"""

# I/O 路径默认值
DEFAULT_SOURCE_DIR: Optional[str] = None
DEFAULT_TARGET_DIR: Optional[str] = "dataset/audio_split"
DEFAULT_OUTPUT_FILE: Optional[str] = "dataset/audio_list/list.txt"

# 处理参数
DEFAULT_CACHE_DIR: str = "cache"
DEFAULT_SAMPLE_RATE: int = 16000
DEFAULT_MAX_SECONDS: int = 12
DEFAULT_USE_ABSOLUTE_PATH: bool = True
DEFAULT_KEEP_CACHE: bool = False

# 分段/合并默认参数
DEFAULT_MERGE_SILENCE_MS: int = 300   # 合并相邻VAD段时允许的最大静音间隔(ms)
DEFAULT_VAD_MAX_SINGLE_SEGMENT_MS: int = 12000  # VAD单段最大时长(ms)

# 外部依赖
FFMPEG_BIN: str = "ffmpeg"
SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.aac')

# 模型目录
ASR_PARAFORMER_DIR = "models/asr_models/paraformer"
PUNC_MODEL_DIR = "models/punc_models/punc_ct"
SPK_MODEL_DIR = "models/spk_models/campplus_sv"
VAD_MODEL_DIR = "models/vad_models/fsmn_vad"

 

def _ensure_exists(path: str, name: str) -> str:
    """检查模型目录是否存在"""
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"未找到{name}模型目录: {path}")

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

## 处理流程：使用组合模型（内部VAD+ASR+PUNC+SPK）直接获得句级时间戳并切片识别

class FunASRProcessor:
    """FunASR处理器 - Paraformer + 内部VAD + PUNC + SPK（单阶段：时间戳驱动切片）"""

    def __init__(self, vad_max_single_segment_ms: int = DEFAULT_VAD_MAX_SINGLE_SEGMENT_MS):
        self.model = None
        self.vad_max_single_segment_ms = int(vad_max_single_segment_ms)
        self.init_model()
    
    def init_model(self):
        """初始化FunASR模型"""
        try:
            # Paraformer 完整组合（ASR + VAD + PUNC + SPK）
            asr_model = _ensure_exists(ASR_PARAFORMER_DIR, "ASR(Paraformer)")
            punc_model = _ensure_exists(PUNC_MODEL_DIR, "PUNC(CT)")
            spk_model = _ensure_exists(SPK_MODEL_DIR, "SPK(CampPlus)")
            vad_model = _ensure_exists(VAD_MODEL_DIR, "VAD(FSMN)")
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                self.model = AutoModel(
                    model=asr_model,
                    vad_model=vad_model,
                    vad_kwargs={"max_single_segment_time": self.vad_max_single_segment_ms},
                    punc_model=punc_model,
                    spk_model=spk_model,
                    disable_update=True,
                )
            
            
        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> List[Dict]:
        """转录音频文件"""
        try:
            # 使用FunASR进行转录（Paraformer组合，返回句级时间戳）
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                result = self.model.generate(input=audio_path, cache={})
            
            # 解析结果
            segments = []
            if isinstance(result, list) and len(result) > 0:
                res = result[0]
                
                # 处理输出格式
                if 'sentence_info' in res:
                    # Paraformer模型输出格式
                    for sentence in res['sentence_info']:
                        segments.append({
                            'start': sentence['start'] / 1000.0,  # 转换为秒
                            'end': sentence['end'] / 1000.0,
                            'text': sentence['text']
                        })
                elif 'text' in res:
                    # 兜底：只有文本没有时间戳
                    text = res['text']
                    
                    if text.strip():
                        segments.append({
                            'start': 0.0,
                            'end': self._get_audio_duration(audio_path),
                            'text': text
                        })
                else:
                    # 最后兜底处理
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
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except:
            return 10.0  # 默认10秒
    
    def __call__(self, audio_in: str) -> List[Dict]:
        """调用接口"""
        return self.transcribe(audio_in)

def init_asr_model(vad_max_single_segment_ms: int):
    """初始化ASR模型（Paraformer：内部VAD+ASR+PUNC+SPK）"""
    return FunASRProcessor(vad_max_single_segment_ms=vad_max_single_segment_ms)

## 单阶段流程不再需要独立VAD模型

## 单阶段流程不再需要独立VAD分段与合并逻辑

def ffmpeg_extract_segment(source_file: str, target_file: str, start_ms: int, end_ms: int, sample_rate: int) -> bool:
    """使用ffmpeg按起止毫秒切片，输出目标采样率单声道wav"""
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    duration_s = max(0.0, (end_ms - start_ms) / 1000.0)
    start_s = max(0.0, start_ms / 1000.0)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start_s}",
        "-t", f"{duration_s}",
        "-i", source_file,
        "-ar", f"{sample_rate}",
        "-ac", "1",
        "-v", "quiet",
        target_file,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"切片失败 {source_file} [{start_ms},{end_ms}]ms: {e}")
        return False

def _segments_to_text(segments: List[Dict]) -> str:
    """将转录分段合并为单条文本"""
    parts: List[str] = []
    for seg in segments or []:
        t = str(seg.get('text', '')).strip()
        if t:
            parts.append(t)
    return "".join(parts).strip()

def recognize_with_internal_timestamps(
    converted_files: List[str], target_dir: str, sample_rate: int,
    asr_model: FunASRProcessor,
    absolute_path: bool,
) -> List[str]:
    """单阶段：直接用组合模型拿句级时间戳并切片识别，返回 list.txt 行列表"""
    results: List[str] = []
    for audio_path in tqdm(converted_files, desc="内部VAD时间戳切片+识别"):
        try:
            segs = asr_model(audio_in=audio_path)
            if not segs:
                continue
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            count = 0
            for seg in segs:
                start_s = float(seg.get('start', 0.0) or 0.0)
                end_s = float(seg.get('end', 0.0) or 0.0)
                text = str(seg.get('text', '') or '').strip()
                if not text:
                    continue
                start_ms = int(max(0.0, start_s) * 1000)
                end_ms = int(max(0.0, end_s) * 1000)
                if end_ms <= start_ms:
                    continue
                sliced_audio_name = f"{audio_name}_{str(count).zfill(6)}"
                sliced_audio_path = os.path.join(target_dir, sliced_audio_name + ".wav")
                ok = ffmpeg_extract_segment(audio_path, sliced_audio_path, start_ms, end_ms, sample_rate)
                if not ok:
                    count += 1
                    continue
                out_path = os.path.abspath(sliced_audio_path) if absolute_path else sliced_audio_path
                results.append(f"{out_path}|{text}")
                count += 1
        except Exception as e:
            print(f"处理错误 {audio_path}: {e}")
            continue
    return results

## 优化的两阶段处理模式：粗分段 + 精细识别

def clean_cache_directory(cache_dir: str):
    """清除缓存目录"""
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception as e:
        print(f"清理缓存失败: {e}")

def create_list(source_dir: str, target_dir: str, cache_dir: str, sample_rate: int,
               output_list: str, max_seconds: int, absolute_path: bool,
               clean_cache: bool = True,
               merge_silence_ms: int = DEFAULT_MERGE_SILENCE_MS,
               vad_max_single_segment_ms: int = DEFAULT_VAD_MAX_SINGLE_SEGMENT_MS):
    """创建训练列表文件"""
    
    resample_dir = os.path.join(cache_dir, "funasr_cache", "origin", f"{sample_rate}")
    
    try:
        print("转换音频文件...")
        converted_files = convert_files(source_dir, resample_dir, sample_rate)
        
        if not converted_files:
            print("错误: 没有找到可处理的音频文件")
            return
        # 单阶段：直接使用组合模型（内部VAD）识别并依据时间戳切片
        print("初始化组合ASR模型（内部VAD+PUNC+SPK）...")
        asr_model = init_asr_model(vad_max_single_segment_ms)
        os.makedirs(target_dir, exist_ok=True)
        print("直接基于内部VAD时间戳进行切片与识别...")
        result = recognize_with_internal_timestamps(
            converted_files, target_dir, sample_rate,
            asr_model,
            absolute_path,
        )
        
        print("保存结果...")
        os.makedirs(os.path.dirname(output_list) if os.path.dirname(output_list) else '.', exist_ok=True)
        
        with open(output_list, "w", encoding="utf-8") as file:
            for line in result:
                file.write(line.strip() + '\n')
        
        print(f"完成! 生成 {len(result)} 条训练数据")
        print(f"输出文件: {output_list}")
        
    finally:
        if clean_cache:
            clean_cache_directory(cache_dir)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用FunASR创建语音训练数据集")
    
    parser.add_argument("source_dir", nargs='?', default=DEFAULT_SOURCE_DIR, type=str, help="源音频目录路径")
    parser.add_argument("target_dir", nargs='?', default=DEFAULT_TARGET_DIR, type=str, help="目标数据集目录路径")
    parser.add_argument("output", nargs='?', default=DEFAULT_OUTPUT_FILE, type=str, help="输出列表文件路径")

    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help=f"缓存目录路径，默认: {DEFAULT_CACHE_DIR}")
    parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help=f"采样率，默认: {DEFAULT_SAMPLE_RATE}")
    parser.add_argument("--max_seconds", type=int, default=DEFAULT_MAX_SECONDS, help=f"最大输出片段长度(秒)，默认: {DEFAULT_MAX_SECONDS}")
    parser.add_argument("--relative_path", action="store_true", help="使用相对路径")
    parser.add_argument("--keep_cache", action="store_true", default=DEFAULT_KEEP_CACHE, help="保留缓存文件")
    parser.add_argument("--merge_silence_ms", type=int, default=DEFAULT_MERGE_SILENCE_MS, help=f"[单阶段无效] 合并相邻VAD段静音(ms)，默认: {DEFAULT_MERGE_SILENCE_MS}")
    parser.add_argument("--vad_max_single_segment_ms", type=int, default=DEFAULT_VAD_MAX_SINGLE_SEGMENT_MS, help=f"VAD模型单段最大时长(ms)，默认: {DEFAULT_VAD_MAX_SINGLE_SEGMENT_MS}")

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
        print("  python datasets_list_create.py /path/to/source_audio")
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
    
    try:
        create_list(
            source_dir, target_dir, cache_dir, 
            args.sample_rate, output_file, 
            # 默认使用绝对路径；如需默认相对路径可在顶部改 DEFAULT_USE_ABSOLUTE_PATH
            args.max_seconds, DEFAULT_USE_ABSOLUTE_PATH and (not args.relative_path),
            not args.keep_cache,  # 默认清理缓存，除非指定保留
            merge_silence_ms=args.merge_silence_ms,
            vad_max_single_segment_ms=args.vad_max_single_segment_ms,
        )
        return 0
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 
