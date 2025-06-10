from faster_whisper import WhisperModel, BatchedInferencePipeline

model_size = "/home/lf/Model/ASR/faster_whisper_models/large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe("/home/lf/音乐/asr_example_en.wav", beam_size=5)
segments, info = batched_model.transcribe("/home/lf/音乐/诗朗诵_我是湖北人.mp3", batch_size=64)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))