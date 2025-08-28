from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

model_dir = "models/asr_models/sensevoice"

# en 
en_input = f"{model_dir}/example/en.mp3"
# zh
zh_input = f"{model_dir}/example/zh.mp3"

input_wav = "/Users/boom/Music/乐飞航空/manual/data/ytc/manual171_0113.wav"

input_dic = {
    "en": en_input,
    "zh": zh_input,
    "zh": input_wav,
}

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model=model_dir,
    )

for input_wav in input_dic.values():
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        res = model.generate(
            input=input_wav,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            output_timestamp=True,
        )
    print(res)
    print(res[0]["text"])
