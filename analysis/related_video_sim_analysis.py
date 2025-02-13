import numpy as np
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt  # 시각화 추가

# ✅ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 데이터 파일 로드
text_features = torch.tensor(np.load("text_features.npy", mmap_mode="r"), dtype=torch.float32).to(device)
video_features = torch.tensor(np.load("video_features.npy", mmap_mode="r"), dtype=torch.float32).to(device)

# ✅ 비디오 및 텍스트 메타데이터 로드
with open("video_metadata.json", "r") as f:
    video_metadata = json.load(f)
with open("text_metadata.json", "r") as f:
    text_metadata = json.load(f)

# ✅ 하이퍼파라미터
text_batch_size = 100
video_batch_size = 5000

# ✅ 정답 비디오 매핑
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

# ✅ 비디오 ID 리스트
video_ids = list(video_metadata.keys())

# ✅ 유사도 저장 리스트
related_video_similarities = []
unrelated_video_similarities = []

# ✅ 모든 텍스트를 미니배치 단위로 처리
print("🔹 Processing Text Batches...")
text_progress = tqdm(range(0, len(text_metadata), text_batch_size), desc="Text Batches", unit="batch")

for text_start in text_progress:
    text_end = min(text_start + text_batch_size, len(text_metadata))

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

    # ✅ 연관된 비디오 및 연관되지 않은 비디오 유사도 병렬 연산
    for i, query_id in enumerate(query_ids):
        gt_vid = query_to_gt[query_id]

        # 🔹 (1) 연관된 비디오 유사도 계산
        if gt_vid in video_ids:
            gt_vid_index = video_ids.index(gt_vid)  # ✅ 문자열 기반으로 인덱스 찾기
            related_video_similarities.append(video_similarities_per_query[i, gt_vid_index].item())

        # 🔹 (2) 연관되지 않은 비디오 유사도 병렬 연산
        unrelated_indices = [idx for idx, vid in enumerate(video_ids) if vid != gt_vid]  # ✅ 문자열 비교
        unrelated_video_sims = video_similarities_per_query[i, unrelated_indices]  # ✅ 연관되지 않은 비디오 선택
        
        if unrelated_video_sims.numel() > 0:  # 비어있지 않은 경우만 평균 계산
            unrelated_video_similarities.append(torch.mean(unrelated_video_sims).item())

# ✅ 전체 통계 출력
print("\n✅ Similarity Distribution Summary:")
print(f"📊 Mean similarity for related videos: {np.mean(related_video_similarities):.4f}")
print(f"📊 Mean similarity for unrelated videos: {np.mean(unrelated_video_similarities):.4f}")
print(f"📉 Min similarity for related videos: {np.min(related_video_similarities):.4f}")
print(f"📉 Min similarity for unrelated videos: {np.min(unrelated_video_similarities):.4f}")
print(f"📈 Max similarity for related videos: {np.max(related_video_similarities):.4f}")
print(f"📈 Max similarity for unrelated videos: {np.max(unrelated_video_similarities):.4f}")

# ✅ 히스토그램 시각화
plt.figure(figsize=(10, 6))
plt.hist(related_video_similarities, bins=50, alpha=0.7, label="Related Videos", color="blue", edgecolor="black")
plt.hist(unrelated_video_similarities, bins=50, alpha=0.7, label="Unrelated Videos", color="red", edgecolor="black")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.title("Distribution of Query-Video Similarities")
plt.savefig("query_video_similarity_distribution.png")

# ✅ JSON 파일 저장
results = {
    "related_video_similarities": related_video_similarities,
    "unrelated_video_similarities": unrelated_video_similarities,
}
with open("query_video_similarity_distribution.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Results saved to query_video_similarity_distribution.json and query_video_similarity_distribution.png")

