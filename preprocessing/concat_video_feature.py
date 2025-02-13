import numpy as np
import os
import json
from tqdm import tqdm

def load_and_average_features(json_path, feature_dir, save_path):
    """
    Load video features stored as .npy files, average them over the first dimension,
    and stack them together.
    
    Args:
        feature_dir (str): Directory containing .npy files, each representing a video's features.

    Returns:
        np.ndarray: Stacked feature array of shape (N * # of Frames, feature dimension).
    """
    # JSON 파일 로드
    with open(json_path, "r") as f:
        video_metadata = json.load(f)

    all_features = []
    # JSON 파일의 key 순서대로 처리
    for video_id in tqdm(video_metadata.keys(), desc="Processing videos"):
        file_name = f"{video_id}.npy"
        file_path = os.path.join(feature_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {feature_dir}, skipping...")
            continue
    
        # Numpy 파일 로드
        features = np.load(file_path)  # Shape: (# of Frames, # of learned queries, feature dimension)
        
        # 두 번째 차원(learned queries)에 대해 평균 계산
        averaged_features = np.mean(features, axis=1)  # Shape: (# of Frames, feature dimension)
        
        all_features.append(averaged_features)

    if not all_features:
        print("No valid features loaded. Exiting...")
        return None
    
    # file_list = sorted(os.listdir(feature_dir))
    # for file_name in tqdm(file_list, desc="Processing files"):
    #     if file_name.endswith(".npy"):
    #         file_path = os.path.join(feature_dir, file_name)
    #         features = np.load(file_path)  # Shape: (# of Frames, # of learned queries, feature dimension)
            
    #         # Average over the second dimension
    #         averaged_features = np.mean(features, axis=1)  # Shape: (# of Frames, feature dimension)
            
    #         all_features.append(averaged_features)
    
    # Stack all features from different videos
    final_features = np.vstack(all_features)  # Shape: (N * # of Frames, feature dimension)

    # Save the final features as a .npy file
    np.save(save_path, final_features)
    
    return final_features

# Example usage
json_path = "dataset/charades-sta/llm_outputs.json"
feature_directory = "datasets/Charades"
save_file = "averaged_charades_features.npy"
averaged_features = load_and_average_features(json_path, feature_directory, save_file)