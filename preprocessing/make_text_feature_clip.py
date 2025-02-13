import json
import torch
import numpy as np
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import clip

#### CLIP ####
clip_model, preprocess = clip.load("ViT-L/14", device='cuda')
clip_text_encoder = clip_model.encode_text
#### CLIP ####

# JSON 파일 로드
with open("../../TAG/dataset/charades-sta/llm_outputs.json", "r") as f:
    data = json.load(f)

# 저장할 데이터 리스트
text_features_list = []
text_metadata = {}

# 데이터 전처리
for vid, details in tqdm(data.items(), desc="Processing Videos", unit="video"):
    sentences = details["sentences"]
    timestamps = details["timestamps"]  # 해당 문장의 후보 구간 정보

    for i, (sentence, timestamp) in enumerate(zip(sentences, timestamps)):
        query_id = f"{vid}_{i}"  # query ID 생성 (vid_0, vid_1, ...)
        with torch.no_grad():
            text_tokens = clip.tokenize(sentences).to(device='cuda')
            text_feat = clip_text_encoder(text_tokens)
            v1 = F.normalize(text_feat, p=2, dim=1)

        # 데이터 저장
        text_features_list.append(v1.cpu().numpy())  
        text_metadata[query_id] = {
            "text": sentence, 
            "vid": vid,
            "timestamps": timestamp  # 후보 구간 정보 추가
        }

# NumPy 배열로 변환 및 저장
text_features_array = np.vstack(text_features_list)  # (num_texts, feature_dim)
np.save("text_features_charades.npy", text_features_array)

# JSON 파일 저장
with open("text_metadata_charades.json", "w") as f:
    json.dump(text_metadata, f, indent=4)

print("✅ text_features.npy & text_metadata.json 생성 완료!")
