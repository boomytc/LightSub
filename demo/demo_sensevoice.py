from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

model_dir = "models/asr_models/sensevoice"

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

# 输出内容如下
'''
[{'key': 'en', 'text': '<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.', 'timestamp': [[0, 1110], [1110, 1710], [1710, 2370], [2370, 2610], [2610, 2850], [2850, 3090], [3090, 3270], [3270, 3870], [3870, 4290], [4290, 4650], [4650, 4830], [5070, 5190], [5190, 5430], [5430, 5730], [5730, 6030], [6030, 6270], [6270, 7170]]}]
<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.
[{'key': 'manual171_0113', 'text': '<|zh|><|NEUTRAL|><|Speech|><|withitn|>增速阶段的注意力分配以外为主，余光兼顾座舱。', 'timestamp': [[0, 690], [690, 930], [930, 1050], [1050, 1230], [1230, 1410], [1410, 1530], [1530, 1710], [1710, 1890], [1890, 2070], [2070, 2250], [2250, 2490], [2490, 2730], [2730, 2910], [2910, 3090], [3090, 3330], [3330, 3450], [3450, 3630], [3630, 3810], [3810, 3990], [3990, 4170], [4170, 4290], [4290, 4650]]}]
<|zh|><|NEUTRAL|><|Speech|><|withitn|>增速阶段的注意力分配以外为主，余光兼顾座舱。
'''