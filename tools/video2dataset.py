import numpy as np
import cv2
#import json
from datetime import datetime
import os
import multiprocessing
import sys
import h5py
import jsonlines as jsonl
sys.path.append("../models/")
sys.path.append( "../models/pose_detection/engines/yolov7_pose/")
from pose_detection.engines import YoloV7 as PoseEngine
#from pose_detection.engines import Movenet as PoseEngine
from hdf5_utils import save_dict_to_hdf5 #save_dict_to_hdf5, load_dict_to_hdf5

def create_folder_tree(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def extract_single_video(filepath):
    if filepath[0].endswith(".avi"):
        total_frames = filepath[2]
        # -- load pose engine
        pose_engine = PoseEngine.Engine()
        # -- open video
        cap = cv2.VideoCapture(filepath[0])
        # -- get frame interval
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(num_frames / total_frames)
        frame_container = []
        kp_container = []
        if frame_interval > 0:
        # -- extract the frames
            for i in range(total_frames):
                frame_num = i * frame_interval
                # -- set the video frame position to the desired frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # -- read the frame from the video
                ret, frame = cap.read()
                if ret == True:
                    keypoints = pose_engine.run(frame)
                    #kp_container.append(keypoints.tolist())
                    #frame_container.append(frame.tolist())
                    kp_container.append(keypoints)
                    frame_container.append(frame)
            kp_container = np.stack(kp_container, axis=0).astype(np.float32)
            frame_container = np.stack(frame_container, axis=0).astype(np.float32)
            json_array = {"file_name": filepath[1], "img_data": np.expand_dims(frame_container, axis=0), "kp_data": np.expand_dims(kp_container, axis=0)}
            return json_array

def get_video_props(file_path, total_frames):
        # -- open first video
        cap = cv2.VideoCapture(file_path)
        # -- get properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # -- release video capture
        cap.release()

        return {"frame_count": frame_count, "fps": fps, "frames_saved": total_frames, "width": width,  "height": height}


if __name__== '__main__':
    # -- get videos to work with
    if sys.argv[1] =="unlabeled_videos":
        folder_src = "unlabeled_videos"
    elif sys.argv[1] == "videos2label":
        folder_src = "videos2label"
    else:
        folder_src = None 
        print("Error: arg options are unlabeled_videos or videos2label")

    # -- dummy model to pre-download weights if they dont exist   
    _model =  PoseEngine.Engine()
    del _model

    if folder_src != None:
        # -- check if output folder exists or create
        create_folder_tree("./datasets/unlabeled_datasets/")
        # -- output file
        output_dir = "./datasets/unlabeled_datasets/" + folder_src + str(round(datetime.now().timestamp())) 
        # -- frames to extract
        total_frames = 20 
        # -- the dictionary
        entries = {"video_name": "", "data": 0}
        # -- create mutiprocessing pool
        pool = multiprocessing.Pool(processes=16)
        print("-- extracting videos --")
        # -- file list
        dir = "./datasets/raw_videos/" + folder_src
        files = os.listdir(dir)
        files_fullpath = [[os.path.join(dir, filename), filename, total_frames] for filename in files]
        # -- process data
        dataset = pool.map(extract_single_video, files_fullpath)
        # -- merge to global dataset
        print(f"The final dataset shape: {np.array(dataset).shape}")
        props = get_video_props(files_fullpath[0][0], total_frames)
        dataset = {"props": props, "dataset": dataset}
        print(dataset.keys())
        # -- close pool
        pool.close()
        # -- save dataset 
        save_dict_to_hdf5(output_dir, dataset)