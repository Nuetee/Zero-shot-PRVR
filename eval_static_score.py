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

# ✅ 비디오별 후보 구간 정보 저장
# video_segments = {vid: torch.tensor(meta["scene_segments"][:-1], device=device) for vid, meta in video_metadata.items()}
video_segments = {vid: torch.tensor(meta["proposals"], device=device) for vid, meta in video_metadata.items()}

# ✅ 하이퍼파라미터
text_batch_size = 100
video_batch_size = 5000
top_k_list = [1, 10, 100]

# ✅ 정답 비디오 매핑
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

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

    # 🔹 유사도 행렬 초기화
    similarity_matrix = torch.zeros((text_batch.shape[0], video_features.shape[0]), device=device)

    # 🔹 비디오 프레임을 미니배치 단위로 연산
    for video_start in range(0, video_features.shape[0], video_batch_size):
        video_end = min(video_start + video_batch_size, video_features.shape[0])
        video_batch = video_features[video_start:video_end].to(device)

        # 🔹 코사인 유사도 계산 (행렬 연산)
        similarity_matrix[:, video_start:video_end] = F.cosine_similarity(
            text_batch.unsqueeze(1), video_batch.unsqueeze(0), dim=2
        )

    # ✅ 모든 query에 대해 비디오별 최고 후보 구간 유사도 계산 (벡터 연산)
    video_ids = list(video_segments.keys())
    num_videos = len(video_ids)

    # ✅ (batch_size, num_videos) 크기의 텐서로 초기화
    best_video_scores_tensor = torch.full((similarity_matrix.shape[0], num_videos), -float("inf"), device=device)

    for vid_idx, vid in enumerate(video_ids):
        if vid not in video_metadata:
            continue  # 메타데이터에 없는 비디오는 스킵

        global_start_idx = video_metadata[vid]["start_index"]
        total_vid_start = global_start_idx
        total_vid_end = global_start_idx + video_metadata[vid]["scene_segments"][-1][0]

        # ✅ 모든 query에 대해 해당 비디오의 similarity_vector 추출
        video_similarities = similarity_matrix[:, total_vid_start:total_vid_end]

        # ✅ cumsum을 한 번만 계산 (전체 query에 대해)
        padded_video_similarities = torch.cat((torch.zeros((video_similarities.shape[0], 1), device=device), video_similarities), dim=1)
        cumsum_sim = torch.cumsum(padded_video_similarities, dim=1)

        total_sum = torch.sum(video_similarities, dim=1, keepdim=True)
        total_frames = video_similarities.shape[1]
        total_avg_similarity = total_sum / total_frames

        # ✅ 비디오 내 segment 상대적인 인덱스로 변환
        segments = video_segments[vid]
        segment_starts = segments[:, 0].unsqueeze(0)  # (1, num_segments)
        segment_ends = segments[:, 1].unsqueeze(0)  # (1, num_segments)

        # ✅ 모든 query에 대해 segment 내부 유사도 평균 계산
        # ✅ segment_starts와 segment_ends가 같은 경우 예외처리
        zero_length_mask = (segment_ends == segment_starts)

        # ✅ zero-length segment에서 segment_starts > 0인지 여부에 따라 다르게 처리
        segment_sums_zero_length = torch.where(
            segment_starts > 0,
            cumsum_sim[:, segment_ends] - cumsum_sim[:, segment_starts - 1],  # segment_starts > 0
            cumsum_sim[:, torch.clamp(segment_ends + 1, max=cumsum_sim.shape[1] - 1)] - cumsum_sim[:, segment_starts]  # segment_starts == 0
        )

        # ✅ 일반적인 segment_sums 계산 (zero-length 아닌 경우)
        segment_sums_normal = cumsum_sim[:, segment_ends] - cumsum_sim[:, segment_starts]

        # ✅ 최종 segment_sums 할당 (zero-length이면 예외 처리된 값 사용)
        segment_sums = torch.where(zero_length_mask, segment_sums_zero_length, segment_sums_normal)

        # ✅ segment_lengths에서 0 방지 (0이면 1로 설정)
        segment_lengths = (segment_ends - segment_starts).float()
        segment_lengths = torch.where(zero_length_mask, torch.ones_like(segment_lengths), segment_lengths)
        segment_avg_similarities = segment_sums / segment_lengths
        
        # ✅ segment 외부 유사도 평균 계산
        non_segment_lengths = total_frames - segment_lengths
        non_segment_sums = total_sum.unsqueeze(-1) - segment_sums
        non_segment_avg_similarities = torch.where(
            non_segment_lengths > 0, non_segment_sums / non_segment_lengths, torch.zeros_like(non_segment_lengths)
        )

        # ✅ 최종 Segment Score 계산 (벡터 연산)
        segment_scores = segment_avg_similarities - non_segment_avg_similarities

        # ✅ 최적 segment 선택 (각 query마다 가장 높은 segment score 선택)
        max_segment_scores = torch.max(segment_scores, dim=-1).values  # (batch_size,)

        # ✅ best_video_scores_tensor에 저장
        best_video_scores_tensor[:, vid_idx] = max_segment_scores.squeeze(-1)  

    # ✅ GPU에서 바로 Top-K 비디오 추출
    _, topk_indices = torch.topk(best_video_scores_tensor, max(top_k_list), dim=1)

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
