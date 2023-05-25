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
#from pose_detection.engines import YoloV7 as PoseEngine
from pose_detection.engines import Movenet as PoseEngine
import hdf5_utils#save_dict_to_hdf5, load_dict_to_hdf5

def extract_single_video(filepath):
    # -- load pose engine
    pose_engine = PoseEngine.Movenet()
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
                kp_container.append(keypoints.tolist())
                frame_container.append(frame.tolist())
    json_array = {"file_name": filepath[1], "data": frame_container, "kp": kp_container}
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
    folder_src = None 
    if sys.argv[1] =="unlabeled_videos":
        folder_src = "unlabeled_videos"
    elif sys.argv[1] == "videos2label":
        folder_src = "videos2label"
    else:
        print("Error: arg options are unlabeled_videos or videos2label")
    # match sys.argv[1]:
    #     case "unlabeled_videos":
    #         folder_src = "unlabeled_videos"
    #     case "videos2label":
    #         folder_src = "videos2label"
    #     case _:
    #         print("Error: arg options are unlabeled_videos or videos2label")

    if folder_src != None:
        # -- output file
        output_dir = f"./datasets/unlabeled_datasets/unlabeled_dataset_{str(round(datetime.now().timestamp()))}" 
        # -- frames to extract
        total_frames = 20 
        # -- the dictionary
        entries = {"video_name": "", "data": 0}
        # -- create mutiprocessing pool
        pool = multiprocessing.Pool(processes=1)
        # -- file list
        dir = "./datasets/raw_videos/" + folder_src
        files = os.listdir(dir)
        files_fullpath = [[os.path.join(dir, filename), filename] for filename in files]
        # -- process data
        dataset = pool.map(extract_single_video, files_fullpath)
        print(np.array(dataset).shape)
        props = get_video_props(files_fullpath[0][0], total_frames)
        dataset = {"props": props, "dataset": dataset}
        # -- save dataset 
    with h5py.File(output_dir + '.h5', 'w') as file:
        group = file.create_group('dictionary')

    # Save the nested dictionary recursively
    #save_dict_to_hdf5(file, group, data_dict)
    #    with jsonl.open(output_dir, mode='w') as writer:
    #        writer.write(dataset)



# -- Single thread processing
#if __name__== '__main__':
#    
#    # -- output file
#    output_dir = f"./dataset/original_datasets/unlabeled_dataset_{str(round(datetime.now().timestamp()))}.json" 
#    # -- frames to extract
#    total_frames = 15
#    # -- the dictionary
#    entries = {"video_name": "", "data": 0}
#    # -- datatset
#    dataset = []
#    # -- loop through files in dir
#    for filename in os.listdir("./dataset/raw_videos/"):
#        # -- get full path of file
#        filepath = os.path.join("./dataset/raw_videos/", filename)
#        # -- open video
#        cap = cv2.VideoCapture(filepath)
#        # -- get frame interval
#        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#        frame_interval = int(num_frames / total_frames)
#        frame_container = []
#        if frame_interval > 0:
#            # -- extract the frames
#            for i in range(total_frames):
#                frame_num = i * frame_interval
#                print(frame_num)
#                # -- set the video frame position to the desired frame
#                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                # -- read the frame from the video
#                ret, frame = cap.read()
#                if ret == True:
#                    frame_container.append(frame.tolist())
#
#        print(np.array(frame_container).shape) 
