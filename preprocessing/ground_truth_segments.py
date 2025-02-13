import json
import numpy as np

def add_frame_segments(json_data, num_segments=5):
    for key, value in json_data.items():
        num_frames = value['scene_segments'][-1][0]
        duration = value['duration']
        timestamps = value['timestamps']
        ground_truths = np.round(np.array(timestamps) / duration * num_frames).astype(int)
        ground_truths = np.clip(ground_truths, 0, num_frames).tolist()

        
        value["ground_truth_segments"] = ground_truths
    
    return json_data

with open("../metadata/activitynet/video_metadata.json", "r") as f:
    json_data = json.load(f)

updated_json = add_frame_segments(json_data)

# 결과를 JSON 파일로 저장
with open("../metadata/activitynet/video_metadata_2.json", "w") as f:
    json.dump(updated_json, f, indent=4)
