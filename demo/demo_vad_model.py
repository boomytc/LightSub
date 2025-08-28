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