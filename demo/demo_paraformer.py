from funasr import AutoModel
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

PARAFORMER_MODEL="models/asr_models/paraformer"

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model=PARAFORMER_MODEL,
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
[{'key': 'manual171_0113', 'text': '增 速 阶 段 的 注 意 力 分 配 以 外 为 主 余 光 兼 顾 座 舱', 'timestamp': [[550, 750], [750, 990], [990, 1110], [1110, 1350], [1350, 1470], [1470, 1630], [1630, 1750], [1750, 1950], [1950, 2110], [2110, 2350], [2350, 2510], [2510, 2750], [2770, 2930], [2930, 3170], [3230, 3390], [3390, 3610], [3610, 3810], [3810, 4010], [4010, 4170], [4170, 4465]]}]
'''