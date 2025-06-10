from funasr import AutoModel

PARAFORMER_ZH_MODEL="/home/lf/Model/ASR/paraformer_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PARAFORMER_EN_MODEL="/home/lf/Model/ASR/paraformer_models/speech_paraformer_asr-en-16k-vocab4199-pytorch"
VAD_MODEL="/home/lf/Model/ASR/vad_models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_MODEL="/home/lf/Model/ASR/punc_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
SPK_MODEL="/home/lf/Model/ASR/spk_models/speech_campplus_sv_zh-cn_16k-common"

# asr_model=PARAFORMER_ZH_MODEL
asr_model=PARAFORMER_EN_MODEL

model = AutoModel(
    model=asr_model,
    vad_model=VAD_MODEL,
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model=PUNC_MODEL,
    spk_model=SPK_MODEL,
    disable_update=True,
)

if asr_model == PARAFORMER_ZH_MODEL:
    res = model.generate(
        input = "/home/lf/音乐/诗朗诵_面朝大海春暖花开.wav",
        cache={},
    )
elif asr_model == PARAFORMER_EN_MODEL:
    res = model.generate(
        input = "/home/lf/音乐/asr_example_en.wav",
        cache={},
        pred_timestamp=True,
        return_raw_text=True,
        sentence_timestamp=True,
        en_post_proc=True,
    )

print(res)


""" 像下面一样调用英语模型来得到详细的时间戳
选择英语模型speech_paraformer_asr-en-16k-vocab4199-pytorch
 res = model.generate(
    input="",
    cache={},
    pred_timestamp=True,
    return_raw_text=True,
    sentence_timestamp=True,
    en_post_proc=True,
)

"""