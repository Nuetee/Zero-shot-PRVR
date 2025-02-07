import numpy as np
import torch
import json
import heapq
from tqdm import tqdm

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

# ✅ 비디오별 후보 구간 정보 저장 (vid -> [(start1, end1), (start2, end2), ...])
video_segments = {vid: torch.tensor(meta["segments"], device=device) for vid, meta in video_metadata.items()}

# ✅ 하이퍼파라미터 설정
text_batch_size = 100
video_batch_size = 5000
top_k_list = [1, 10, 100]

# ✅ 정답 비디오 매핑 (query_id -> 정답 비디오 ID)
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

# ✅ Top-K 평가 결과 저장
correct_top_k = {k: 0 for k in top_k_list}
total_queries = len(text_metadata)

# ✅ 모든 텍스트를 미니배치 단위로 처리 (GPU 활용)
print("🔹 Processing Text Batches...")
for text_start in tqdm(range(0, total_queries, text_batch_size), desc="Text Batches"):
    text_end = min(text_start + text_batch_size, total_queries)

    # 🔹 텍스트 배치 불러오기 (GPU 이동)
    text_batch = text_features[text_start:text_end].to(device)
    query_ids = list(text_metadata.keys())[text_start:text_end]

    # 🔹 유사도 행렬 저장 (현재 텍스트 배치에 대한 비디오 유사도)
    similarity_matrix = torch.zeros((text_batch_size, video_features.shape[0]), device=device)

    # 🔹 비디오 프레임도 미니배치 단위로 읽으며 연산 (GPU 활용)
    for video_start in tqdm(range(0, video_features.shape[0], video_batch_size), desc="Video Batches", leave=False):
        video_end = min(video_start + video_batch_size, video_features.shape[0])

        # 🔹 비디오 프레임 배치 불러오기 (GPU 이동)
        video_batch = video_features[video_start:video_end].to(device)

        # 🔹 현재 텍스트 배치와 비디오 프레임 배치 간 유사도 계산 (GPU에서 행렬곱 수행)
        similarity_matrix[:, video_start:video_end] = text_batch @ video_batch.T

    # 🔹 각 텍스트별 Top-K 비디오 찾기
    for i, query_id in tqdm(enumerate(query_ids), desc="Processing Queries", leave=False, total=len(query_ids)):
        similarity_vector = similarity_matrix[i]

        # 🔹 비디오별 최고 후보 구간 유사도 찾기 (벡터 연산으로 변경)
        best_video_scores = {}

        for vid, segments in video_segments.items():
            if vid not in video_metadata:
                continue  # 메타데이터에 없는 비디오는 스킵

            global_start_idx = video_metadata[vid]["start_index"]
            total_vid_start = global_start_idx
            total_vid_end = global_start_idx + video_metadata[vid]["scene_segments"][-1][0]

            # ✅ 비디오 전체 평균 유사도 계산
            video_similarities = similarity_vector[total_vid_start:total_vid_end]
            padded_video_similarities = torch.cat((torch.tensor([0.0], device=video_similarities.device), video_similarities))
            cumsum_sim = torch.cumsum(padded_video_similarities, dim=0)  # ✅ cumsum을 한 번만 계산
            
            total_sum = torch.sum(video_similarities)
            total_frames = video_similarities.numel()
            total_avg_similarity = total_sum / total_frames

            # ✅ Segment별 평균 유사도 벡터 연산
            segment_starts = global_start_idx + segments[:, 0]  # (num_segments,)
            segment_ends = global_start_idx + segments[:, 1]  # (num_segments,)

            # ✅ Segment 내부 유사도 평균 계산
            segment_sums = cumsum_sim[segment_ends] - cumsum_sim[segment_starts]
            segment_lengths = (segment_ends - segment_starts).float()
            segment_avg_similarities = segment_sums / segment_lengths

            # ✅ Segment 외부 유사도 평균 계산
            non_segment_lengths = total_frames - segment_lengths
            non_segment_sums = total_sum - segment_sums
            non_segment_avg_similarities = torch.where(
                non_segment_lengths > 0, non_segment_sums / non_segment_lengths, torch.tensor(0, device=device)
            )

            # ✅ 최종 Segment Score 계산
            segment_scores = segment_avg_similarities - non_segment_avg_similarities

            # ✅ 최적 segment 선택
            max_segment_score = torch.max(segment_scores).item()
            best_video_scores[vid] = max_segment_score

        # ✅ 비디오 정렬 최적화 (heapq.nlargest)
        sorted_videos = heapq.nlargest(max(top_k_list), best_video_scores, key=best_video_scores.get)

        # 🔹 각 top_k에 대해 평가
        for k in top_k_list:
            top_k_videos = sorted_videos[:k]
            if query_to_gt[query_id] in top_k_videos:
                correct_top_k[k] += 1

# ✅ 최종 평가 결과 출력
print("\n✅ Evaluation Results:")
for k in top_k_list:
    accuracy = correct_top_k[k] / total_queries
    print(f"🎯 Top-{k} Accuracy: {accuracy:.4f}")
