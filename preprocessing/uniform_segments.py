import json

def add_frame_segments(json_data, num_segments=5):
    for key, value in json_data.items():
        num_frames = value['scene_segments'][-1][0]
        segment_size = num_frames / num_segments
        
        segments = []
        for i in range(num_segments):
            start_frame = round(i * segment_size)
            end_frame = round((i + 1) * segment_size) if i < num_segments - 1 else num_frames
            segments.append([start_frame, end_frame])
        
        value["uniform_segments"] = segments
    
    return json_data

with open("../metadata/charades-sta/video_metadata.json", "r") as f:
    json_data = json.load(f)

updated_json = add_frame_segments(json_data)

# 결과를 JSON 파일로 저장
with open("../metadata/charades-sta/video_metadata_2.json", "w") as f:
    json.dump(updated_json, f, indent=4)
