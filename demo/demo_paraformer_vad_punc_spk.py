from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

PARAFORMER_MODEL="models/asr_models/paraformer"
VAD_MODEL="models/vad_models/fsmn_vad"
PUNC_MODEL="models/punc_models/punc_ct"
SPK_MODEL="models/spk_models/campplus_sv"

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model=PARAFORMER_MODEL,
        vad_model=VAD_MODEL,
        vad_kwargs={"max_single_segment_time": 30000},
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL,
        # preset_spk_num=2,  # 直接指定说话人数量
        disable_update=True,
    )

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    res = model.generate(
        input = "demo/examples/manual171_0113.wav",
        cache={},
    )

print(res)

# 输出内容如下
'''
[{'key': 'manual171_0113', 'text': '增速阶段的注意力分配以外为主，余光兼顾座舱。', 'timestamp': [[580, 740], [740, 980], [1000, 1160], [1160, 1340], [1340, 1460], [1460, 1620], [1620, 1740], [1740, 1960], [1960, 2100], [2100, 2340], [2340, 2500], [2500, 2740], [2760, 2920], [2920, 3160], [3240, 3380], [3380, 3620], [3620, 3800], [3800, 4000], [4000, 4160], [4160, 4455]], 'sentence_info': [{'text': '增速阶段的注意力分配以外为主，', 'start': 580, 'end': 3160, 'timestamp': [[580, 740], [740, 980], [1000, 1160], [1160, 1340], [1340, 1460], [1460, 1620], [1620, 1740], [1740, 1960], [1960, 2100], [2100, 2340], [2340, 2500], [2500, 2740], [2760, 2920], [2920, 3160]], 'spk': 0}, {'text': '余光兼顾座舱。', 'start': 3240, 'end': 4455, 'timestamp': [[3240, 3380], [3380, 3620], [3620, 3800], [3800, 4000], [4000, 4160], [4160, 4455]], 'spk': 0}]}]
'''