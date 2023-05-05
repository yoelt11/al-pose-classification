#Tools

This directory contains several tools useful for the creation and testing of datasets

##Video Stream (Client)
The "video_stream.py" is a script meant to be deployed in the device holding the camera and it sends the image via a websocket stream.
**Args:** target_websocket_address. E.g., 10.0.0.0

##Record Video (Server)
The "record_video.py" is a websocket server that receive images from a video clint and records them to a file located in the "datasets/raw_videos"

##Video to Dataset
The "video2dataset.py" extracts a certain number of frames from the videos located in the "dataset/raw_vides" and saves them to the "unlabeled_datasets". The unlabeled dataset is a json file with the following structure:
```json
{
    "props": {
            "frame_count": frame_count,
            "fps": fps
            "frames_saved": total_frames 
    },
        "dataset": [{
            "file_name": file_name,
            "data": data
        }]
}
```
##Dataset Labelers
The "dataset_labeler.py" allows to play each video from the unlabeled_dataset folder and label via an input prompt, thus making it easy to label a lot of videos. This files are sabed in "dataset(labeled_data)"

##Export to keypoints
The "export2keypoints.py" allows to extract the keypoints using a pose_detection model from an entry of the labeled_data or unlabeled_data directories
