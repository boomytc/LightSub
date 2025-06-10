import os
from faster_whisper import WhisperModel, BatchedInferencePipeline

def load_model():
    """加载模型并保持在显存中"""
    model_size = "/home/lf/Model/ASR/faster_whisper_models/large-v3"

    print("正在加载模型...")
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)

    # 其他可选配置:
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")  # GPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")  # CPU with INT8

    print("模型加载完成！")
    return model, batched_model

def transcribe_audio(model, batched_model, audio_path, use_batched=True):
    """转录音频文件"""
    if not os.path.exists(audio_path):
        print(f"错误：文件 '{audio_path}' 不存在")
        return

    print(f"正在转录: {audio_path}")

    try:
        if use_batched:
            # 使用批处理模式，速度更快
            segments, info = batched_model.transcribe(audio_path, batch_size=64)
        else:
            # 使用普通模式
            segments, info = model.transcribe(audio_path, beam_size=5)

        print(f"检测到语言: '{info.language}' (置信度: {info.language_probability:.2f})")
        print("转录结果:")
        print("-" * 50)

        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        print("-" * 50)

    except Exception as e:
        print(f"转录过程中出现错误: {e}")

def main():
    """主函数 - 交互式音频转录"""
    # 加载模型
    model, batched_model = load_model()

    # 默认使用批处理模式
    use_batched_mode = True

    print("\n=== Faster-Whisper 交互式音频转录 ===")
    print("输入音频文件路径进行转录，输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助信息")
    print(f"当前模式: {'批处理模式' if use_batched_mode else '普通模式'}")
    print("=" * 40)

    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入音频文件路径: ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("退出程序...")
                break

            # 显示帮助信息
            if user_input.lower() == 'help':
                print("\n帮助信息:")
                print("- 输入音频文件的完整路径或相对路径")
                print("- 支持的格式: wav, mp3, mp4, flac, m4a 等")
                print("- 输入 'quit' 或 'exit' 退出程序")
                print("- 输入 'batch' 切换批处理模式")
                print("- 输入 'normal' 切换普通模式")
                continue

            # 检查模式切换
            if user_input.lower() == 'batch':
                use_batched_mode = True
                print("已切换到批处理模式（更快）")
                continue
            elif user_input.lower() == 'normal':
                use_batched_mode = False
                print("已切换到普通模式")
                continue

            # 检查空输入
            if not user_input:
                print("请输入有效的音频文件路径")
                continue

            # 转录音频
            transcribe_audio(model, batched_model, user_input, use_batched_mode)

        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，退出程序...")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()