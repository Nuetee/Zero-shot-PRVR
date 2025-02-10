import numpy as np
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F  # cosine_similarity 사용

# ✅ GPU 설정 (가능하면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 데이터 파일 로드 (mmap_mode="r" 사용하여 메모리에 올리지 않고 읽기)
text_features = torch.tensor(np.load("text_features.npy", mmap_mode="r"), dtype=torch.float32).to(device)
video_features = torch.tensor(np.load("video_features.npy", mmap_mode="r"), dtype=torch.float32).to(device)

# ✅ 비디오 메타데이터 로드
with open("video_metadata.json", "r") as f:
    video_metadata = json.load(f)

# ✅ 텍스트 메타데이터 로드 (query_id, 정답 비디오 정보 포함)
with open("text_metadata.json", "r") as f:
    text_metadata = json.load(f)

# ✅ 비디오별 후보 구간 정보 저장
video_segments = {vid: torch.tensor(meta["proposals"], device=device) for vid, meta in video_metadata.items()}

# ✅ 하이퍼파라미터 설정
text_batch_size = 100
video_batch_size = 5000

# ✅ 정답 비디오 매핑 (query_id -> 정답 비디오 ID)
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

# ✅ 결과 저장용 리스트
results = []

# ✅ 모든 텍스트를 미니배치 단위로 처리 (GPU 활용)
print("🔹 Processing Text Batches...")
text_progress = tqdm(range(0, len(text_metadata), text_batch_size), desc="Text Batches", unit="batch")

for text_start in text_progress:
    text_end = min(text_start + text_batch_size, len(text_metadata))

    # 🔹 텍스트 배치 불러오기 (GPU 이동)
    text_batch = text_features[text_start:text_end].to(device)
    query_ids = list(text_metadata.keys())[text_start:text_end]

    # ✅ 코사인 유사도 계산을 위한 정규화
    text_batch = F.normalize(text_batch, p=2, dim=1)

    # 🔹 유사도 행렬 저장 (현재 텍스트 배치에 대한 비디오 유사도)
    similarity_matrix = torch.zeros((text_batch.shape[0], video_features.shape[0]), device=device)

    # 🔹 비디오 프레임도 미니배치 단위로 읽으며 연산 (GPU 활용)
    for video_start in tqdm(range(0, video_features.shape[0], video_batch_size), desc="Video Batches", leave=False):
        video_end = min(video_start + video_batch_size, video_features.shape[0])

        # 🔹 비디오 프레임 배치 불러오기 (GPU 이동)
        video_batch = video_features[video_start:video_end].to(device)

        # ✅ 코사인 유사도 계산을 위한 정규화
        video_batch = F.normalize(video_batch, p=2, dim=1)

        # ✅ 코사인 유사도 계산
        similarity_matrix[:, video_start:video_end] = F.cosine_similarity(
            text_batch.unsqueeze(1), video_batch.unsqueeze(0), dim=2
        )

    # 🔹 각 쿼리별 분석
    for i, query_id in enumerate(query_ids):
        similarity_vector = similarity_matrix[i]

        # 🔹 정답 비디오 ID 가져오기
        gt_vid = query_to_gt[query_id]
        
        if gt_vid not in video_metadata:
            continue  # 정답 비디오 정보가 없는 경우 스킵

        # 🔹 정답 비디오의 유사도 벡터 가져오기
        global_start_idx = video_metadata[gt_vid]["start_index"]
        total_vid_start = global_start_idx
        total_vid_end = global_start_idx + video_metadata[gt_vid]["scene_segments"][-1][0]

        # ✅ (1) 정답 비디오 전체 프레임 유사도 평균값
        video_similarities = similarity_vector[total_vid_start:total_vid_end]
        mean_video_similarity = torch.mean(video_similarities).item()
        min_video_similarity = torch.min(video_similarities).item()
        max_video_similarity = torch.max(video_similarities).item()

        # ✅ (2) 정답 비디오 내 segment들의 유사도 통계 계산
        if gt_vid in video_segments:
            segments = video_segments[gt_vid]
            padded_video_similarities = torch.cat((torch.tensor([0.0], device=video_similarities.device), video_similarities))
            cumsum_sim = torch.cumsum(padded_video_similarities, dim=0)

            segment_starts = segments[:, 0]  
            segment_ends = segments[:, 1]  

            segment_sums = cumsum_sim[segment_ends] - cumsum_sim[segment_starts]
            segment_lengths = (segment_ends - segment_starts).float()
            segment_avg_similarities = segment_sums / segment_lengths

            min_segment_similarity = torch.min(segment_avg_similarities).item()
            max_segment_similarity = torch.max(segment_avg_similarities).item()
            mean_segment_similarity = torch.mean(segment_avg_similarities).item()
        else:
            min_segment_similarity = None
            max_segment_similarity = None
            mean_segment_similarity = None

        # ✅ 결과 저장
        results.append({
            "query_id": query_id,
            "gt_vid": gt_vid,
            "mean_video_similarity": mean_video_similarity,
            "min_video_similarity": min_video_similarity,
            "max_video_similarity": max_video_similarity,
            "min_segment_similarity": min_segment_similarity,
            "max_segment_similarity": max_segment_similarity,
            "mean_segment_similarity": mean_segment_similarity,
        })

# ✅ 전체 통계 계산
valid_mean_video_sims = [r["mean_video_similarity"] for r in results if r["mean_video_similarity"] is not None]
valid_min_video_sims = [r["min_video_similarity"] for r in results if r["min_video_similarity"] is not None]
valid_max_video_sims = [r["max_video_similarity"] for r in results if r["max_video_similarity"] is not None]
valid_min_segment_sims = [r["min_segment_similarity"] for r in results if r["min_segment_similarity"] is not None]
valid_max_segment_sims = [r["max_segment_similarity"] for r in results if r["max_segment_similarity"] is not None]
valid_mean_segment_sims = [r["mean_segment_similarity"] for r in results if r["mean_segment_similarity"] is not None]

print("\n✅ Overall Results Summary:")
print(f"📊 Mean of mean_video_similarity: {np.mean(valid_mean_video_sims):.4f}" if valid_mean_video_sims else "No valid mean_video_similarity")
print(f"📉 Min of min_video_similarity: {np.min(valid_min_video_sims):.4f}" if valid_min_video_sims else "No valid min_video_similarity")
print(f"📈 Max of max_video_similarity: {np.max(valid_max_video_sims):.4f}" if valid_max_video_sims else "No valid max_video_similarity")
print(f"📊 Mean of mean_segment_similarity: {np.mean(valid_mean_segment_sims):.4f}" if valid_mean_segment_sims else "No valid mean_segment_similarity")
print(f"📉 Min of min_segment_similarity: {np.min(valid_min_segment_sims):.4f}" if valid_min_segment_sims else "No valid min_segment_similarity")
print(f"📈 Max of max_segment_similarity: {np.max(valid_max_segment_sims):.4f}" if valid_max_segment_sims else "No valid max_segment_similarity")