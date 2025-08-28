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

# 输出内容如下
'''
[{'key': 'en', 'text': '<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.', 'timestamp': [[800, 1250], [1250, 1730], [1730, 2390], [2390, 2690], [2690, 2870], [2870, 3050], [3050, 3290], [3290, 3890], [3890, 4310], [4310, 4670], [4670, 4850], [5090, 5210], [5210, 5450], [5450, 5570], [5570, 5990], [5990, 6290], [6290, 7130]]}]
<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.
[{'key': 'manual171_0113', 'text': '<|zh|><|NEUTRAL|><|Speech|><|withitn|>增速阶段的注意力分配以外为主，余光兼顾座舱。', 'timestamp': [[290, 680], [680, 920], [920, 1100], [1100, 1220], [1220, 1400], [1400, 1520], [1520, 1700], [1700, 1880], [1880, 2060], [2060, 2240], [2240, 2540], [2540, 2720], [2720, 2900], [2900, 3080], [3080, 3320], [3320, 3440], [3440, 3620], [3620, 3800], [3800, 3980], [3980, 4160], [4160, 4280], [4280, 4640]]}]
<|zh|><|NEUTRAL|><|Speech|><|withitn|>增速阶段的注意力分配以外为主，余光兼顾座舱。
'''