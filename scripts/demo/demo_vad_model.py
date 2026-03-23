from funasr import AutoModel
import os

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    model = AutoModel(
        model="models/vad_models/fsmn_vad",
        max_single_segment_time=3000, 
        disable_update=True,
    )

wav_file = os.path.join(model.model_path, "example/vad_example.wav")

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    res = model.generate(input=wav_file)

print(res)

# 输出内容如下
'''
[{'key': 'vad_example', 'value': [[70, 2340], [2620, 5630], [5630, 6200], [6480, 9490], [9490, 12500], [12500, 15510], [15510, 18520], [18520, 21530], [21530, 23670], [23950, 26250], [26780, 28990], [29950, 31430], [31750, 34760], [34760, 37770], [38210, 41220], [41220, 44230], [44230, 47240], [47310, 49630], [49910, 52920], [52920, 55930], [55930, 56460], [56740, 59750], [59790, 62800], [62830, 65840], [65840, 68850], [68850, 70450]]}]
'''