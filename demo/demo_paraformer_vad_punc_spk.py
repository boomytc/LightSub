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
        disable_update=True,
    )

with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
    res = model.generate(
        input = "demo/examples/manual171_0113.wav",
        cache={},
    )

print(res)