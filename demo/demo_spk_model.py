import os
import numpy as np
from funasr import AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

spk_model = AutoModel(
    model="models/spk_models/campplus_sv",
    disable_update=True,
)

speaker1_a_wav = 'models/spk_models/campplus_sv/examples/speaker1_a_cn_16k.wav'
speaker1_b_wav = 'models/spk_models/campplus_sv/examples/speaker1_b_cn_16k.wav'
speaker2_a_wav = 'models/spk_models/campplus_sv/examples/speaker2_a_cn_16k.wav'

def extract_speaker_embedding(audio_path):
    """提取说话人embedding"""
    result = spk_model.generate(input=audio_path)
    if isinstance(result, list) and len(result) > 0:
        return result[0].get('spk_embedding', None)
    return None

def compute_similarity(emb1, emb2):
    """计算两个embedding的余弦相似度"""
    if emb1 is None or emb2 is None:
        return 0.0
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.cpu().numpy()
    
    # 确保是2D数组
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def speaker_verification(audio1, audio2, threshold=0.5):
    """说话人验证"""
    emb1 = extract_speaker_embedding(audio1)
    emb2 = extract_speaker_embedding(audio2)
    
    if emb1 is None or emb2 is None:
        return {"score": 0.0, "label": "different", "embeddings": [emb1, emb2]}
    
    score = compute_similarity(emb1, emb2)
    label = "same" if score >= threshold else "different"
    
    return {
        "score": float(score),
        "label": label,
        "embeddings": [emb1, emb2]
    }

print("=== 相同说话人语音 ===")
result = speaker_verification(speaker1_a_wav, speaker1_b_wav)
print(f"相似度得分: {result['score']:.4f}")
print(f"判断结果: {result['label']}")
print()

print("=== 不同说话人语音 ===")
result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
print(f"相似度得分: {result['score']:.4f}")
print(f"判断结果: {result['label']}")
print()

print("=== 自定义阈值 (0.31) ===")
result = speaker_verification(speaker1_a_wav, speaker2_a_wav, threshold=0.31)
print(f"相似度得分: {result['score']:.4f}")
print(f"判断结果: {result['label']}")
print()

print("=== 输出embedding信息 ===")
result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
print(f"相似度得分: {result['score']:.4f}")
print(f"判断结果: {result['label']}")
if result['embeddings'][0] is not None:
    print(f"Speaker 1 embedding shape: {result['embeddings'][0].shape}")
if result['embeddings'][1] is not None:
    print(f"Speaker 2 embedding shape: {result['embeddings'][1].shape}")
print()

print("=== 保存embedding到文件 ===")
save_dir = 'savePath/'
os.makedirs(save_dir, exist_ok=True)

emb1 = extract_speaker_embedding(speaker1_a_wav)
emb2 = extract_speaker_embedding(speaker2_a_wav)

if emb1 is not None:
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.cpu().numpy()
    np.save(os.path.join(save_dir, 'speaker1_embedding.npy'), emb1)
    print(f"Speaker 1 embedding saved to {save_dir}speaker1_embedding.npy")

if emb2 is not None:
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.cpu().numpy()
    np.save(os.path.join(save_dir, 'speaker2_embedding.npy'), emb2)
    print(f"Speaker 2 embedding saved to {save_dir}speaker2_embedding.npy")

print("\n=== FunASR Speaker Verification Demo 完成 ===")