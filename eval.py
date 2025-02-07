import torch
import numpy as np
import json
from tqdm import tqdm

# GPU 설정 (가능하면 GPU 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 파일 로드 (mmap_mode="r"는 CPU에서만 동작하므로 GPU 사용 시 메모리에 로드)
text_features = torch.tensor(np.load("text_features.npy"), dtype=torch.float32).to(device)  # (num_texts, feature_dim)
video_features = torch.tensor(np.load("video_features.npy"), dtype=torch.float32).to(device)  # (total_frames, feature_dim)

# 비디오 메타데이터 로드
with open("video_metadata.json", "r") as f:
    video_metadata = json.load(f)

# 텍스트 메타데이터 로드 (query_id, 정답 비디오 정보 포함)
with open("text_metadata.json", "r") as f:
    text_metadata = json.load(f)

# 비디오별 후보 구간 정보 저장 (vid -> [(start1, end1), (start2, end2), ...])
video_segments = {vid: meta["scene_segments"] for vid, meta in video_metadata.items()}

# 하이퍼파라미터 설정
text_batch_size = 100    # 텍스트 미니배치 크기
video_batch_size = 5000  # 비디오 프레임 미니배치 크기 (너무 크면 메모리 초과)
top_k_list = [1, 10, 100]  # 여러 개의 top-k 평가 수행

# 정답 비디오 매핑 (query_id -> 정답 비디오 ID)
query_to_gt = {qid: meta["vid"] for qid, meta in text_metadata.items()}

# Top-K 평가 결과 저장
correct_top_k = {k: 0 for k in top_k_list}  # {1: 0, 10: 0, 100: 0}
total_queries = len(text_metadata)

# 모든 텍스트를 미니배치 단위로 처리 (GPU 활용)
print("Processing Text Batches...")
for text_start in tqdm(range(0, total_queries, text_batch_size), desc="Text Batches"):
    text_end = min(text_start + text_batch_size, total_queries)

    # 텍스트 배치 불러오기 (GPU 이동)
    text_batch = text_features[text_start:text_end].to(device)  # (text_batch_size, feature_dim)
    query_ids = list(text_metadata.keys())[text_start:text_end]

    # 유사도 행렬 저장 (현재 텍스트 배치에 대한 비디오 유사도)
    similarity_matrix = torch.zeros((text_batch_size, video_features.shape[0]), device=device)

    # 비디오 프레임도 미니배치 단위로 읽으며 연산 (GPU 활용)
    for video_start in range(0, video_features.shape[0], video_batch_size):
        video_end = min(video_start + video_batch_size, video_features.shape[0])
        
        # 비디오 프레임 배치 불러오기 (GPU 이동)
        video_batch = video_features[video_start:video_end].to(device)  # (video_batch_size, feature_dim)

        # 현재 텍스트 배치와 비디오 프레임 배치 간 유사도 계산 (GPU에서 행렬곱 수행)
        similarity_matrix[:, video_start:video_end] = text_batch @ video_batch.T  # (text_batch_size, video_batch_size)
    
    # 각 텍스트별 Top-K 비디오 찾기
    for i, query_id in enumerate(query_ids):
        similarity_vector = similarity_matrix[i]  # (total_frames,)
        
        # 비디오별 최고 후보 구간 유사도 찾기
        best_video_scores = {}

        for vid, segments in video_segments.items():
            max_segment_score = -float("inf")  # 해당 비디오에서 가장 높은 후보 구간 유사도 저장
            global_start_idx = video_metadata[vid]["start_index"]  # 전체 프레임 기준의 시작 인덱스

            for segment in segments:
                local_start, local_end = segment  # 비디오 내 구간
                start_idx = global_start_idx + local_start  # 전체 프레임 기준으로 변환
                end_idx = global_start_idx + local_end      # 전체 프레임 기준으로 변환

                # 범위 초과 방지
                if end_idx > similarity_vector.shape[0]:
                    continue
                
                # 해당 구간의 프레임 유사도 가져오기
                segment_similarities = similarity_vector[start_idx:end_idx]
                avg_similarity = torch.mean(segment_similarities).item()  # 후보 구간 평균 유사도
                
                # 현재 비디오에서 최고 유사도 갱신
                if avg_similarity > max_segment_score:
                    max_segment_score = avg_similarity
            
            # 최종 비디오별 최고 구간 유사도를 저장
            best_video_scores[vid] = max_segment_score
        
        # 유사도 상위 K개 비디오 추출
        sorted_videos = sorted(best_video_scores, key=best_video_scores.get, reverse=True)

        # 각 top_k에 대해 평가
        for k in top_k_list:
            top_k_videos = sorted_videos[:k]
            if query_to_gt[query_id] in top_k_videos:
                correct_top_k[k] += 1

# 최종 평가 결과 출력
for k in top_k_list:
    accuracy = correct_top_k[k] / total_queries
    print(f"Top-{k} Accuracy: {accuracy:.4f}")