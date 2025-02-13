import numpy as np
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import os

# Argument Parsing
parser = argparse.ArgumentParser(description="Evaluate text-video similarity.")
parser.add_argument("--feature_path", type=str, required=True, help="Path to features .npy file")
parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata .json file")
args = parser.parse_args()

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 데이터 파일 로드
text_features = torch.tensor(np.load(os.path.join(args.feature_path, "text_features.npy"), mmap_mode="r"), dtype=torch.float32).to(device)
video_features = torch.tensor(np.load(os.path.join(args.feature_path, "video_features.npy"), mmap_mode="r"), dtype=torch.float32).to(device)

# ✅ 비디오 및 텍스트 메타데이터 로드
with open(os.path.join(args.metadata_path, "video_metadata.json"), "r") as f:
    video_metadata = json.load(f)
with open(os.path.join(args.metadata_path, "text_metadata.json"), "r") as f:
    text_metadata = json.load(f)

# ✅ 하이퍼파라미터
text_batch_size = 100
video_batch_size = 5000
top_k_list = [1, 10, 100]

# ✅ 정답 비디오 매핑 (query_id -> 정답 비디오 ID)
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

# ✅ 비디오 ID 리스트
video_ids = list(video_metadata.keys())

# ✅ Top-K 평가 결과 저장
correct_top_k = {k: 0 for k in top_k_list}
total_queries = len(text_metadata)
processed_queries = 0

# ✅ 모든 텍스트를 미니배치 단위로 처리
print("🔹 Processing Text Batches...")
text_progress = tqdm(range(0, total_queries, text_batch_size), desc="Text Batches", unit="batch")

for text_start in text_progress:
    text_end = min(text_start + text_batch_size, total_queries)

    # 🔹 텍스트 배치 불러오기
    text_batch = text_features[text_start:text_end].to(device)
    query_ids = list(text_metadata.keys())[text_start:text_end]

    # ✅ 코사인 유사도를 위해 정규화
    text_batch = F.normalize(text_batch, p=2, dim=1)

    # 🔹 유사도 행렬 초기화
    similarity_matrix = torch.zeros((text_batch.shape[0], video_features.shape[0]), device=device)

    # 🔹 비디오 프레임을 미니배치 단위로 연산
    for video_start in range(0, video_features.shape[0], video_batch_size):
        video_end = min(video_start + video_batch_size, video_features.shape[0])
        video_batch = video_features[video_start:video_end].to(device)

        # ✅ 코사인 유사도를 위해 정규화
        video_batch = F.normalize(video_batch, p=2, dim=1)

        # ✅ 코사인 유사도 계산 (행렬 연산)
        similarity_matrix[:, video_start:video_end] = F.cosine_similarity(
            text_batch.unsqueeze(1), video_batch.unsqueeze(0), dim=2
        )

    # ✅ 비디오 단위 평균 유사도 계산 (프레임 단위 -> 비디오 단위)
    video_similarities_per_query = torch.zeros((text_batch.shape[0], len(video_ids)), device=device)

    for vid_idx, vid in enumerate(video_ids):
        if vid not in video_metadata:
            continue  # 없는 비디오는 스킵

        global_start_idx = video_metadata[vid]["start_index"]
        total_vid_start = global_start_idx
        total_vid_end = global_start_idx + video_metadata[vid]["scene_segments"][-1][0]

        # ✅ 비디오 내 모든 프레임 유사도의 평균을 구해서 비디오 단위 유사도 생성
        video_similarities_per_query[:, vid_idx] = similarity_matrix[:, total_vid_start:total_vid_end].mean(dim=1)

    # ✅ Top-K 비디오 Retrieval 수행
    _, topk_indices = torch.topk(video_similarities_per_query, max(top_k_list), dim=1)

    # ✅ Top-K 비디오 리스트 변환
    sorted_videos_batch = [[video_ids[idx] for idx in indices] for indices in topk_indices.tolist()]

    # ✅ 평가 수행
    for i, query_id in enumerate(query_ids):
        sorted_videos = sorted_videos_batch[i]
        for k in top_k_list:
            top_k_videos = sorted_videos[:k]
            if query_to_gt[query_id] in top_k_videos:
                correct_top_k[k] += 1

    # ✅ 중간 평가 결과 업데이트
    processed_queries += len(query_ids)
    current_top_k_acc = {k: correct_top_k[k] / processed_queries for k in top_k_list}

    # ✅ tqdm에 정확도 업데이트
    text_progress.set_postfix({
        f"Top-{k} Acc": f"{current_top_k_acc[k]:.4f}" for k in top_k_list
    })

# ✅ 최종 평가 결과 출력
print("\n✅ Evaluation Results:")
for k in top_k_list:
    accuracy = correct_top_k[k] / total_queries
    print(f"🎯 Top-{k} Accuracy: {accuracy:.4f}")
