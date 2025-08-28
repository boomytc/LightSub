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
        input = "/Users/boom/Music/乐飞航空/manual/data/ytc/manual171_0113.wav",
        cache={},
    )

print(res)