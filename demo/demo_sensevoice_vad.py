from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

model_dir = "models/asr_models/sensevoice"

generate_kwargs = {
    "language": "auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    "use_itn": True,
    "batch_size_s": 60,
    "merge_vad": True,  
    "merge_length_s": 15,
}

# en 
en_input = f"{model_dir}/example/en.mp3"
# zh
zh_input = f"{model_dir}/example/zh.mp3"

input_wav = "demo/examples/manual171_0113.wav"

input_dic = {
    "en": en_input,
    "zh": zh_input,
    "zh": input_wav,
}

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model=model_dir,
        vad_model="models/vad_models/fsmn_vad",
        vad_kwargs={"max_single_segment_time": 30000},
    )

for input_wav in input_dic.values():
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        res = model.generate(
            input=input_wav,
            cache={},
            **generate_kwargs,
            output_timestamp=True,
        )
    print(res)
    print(res[0]["text"])
