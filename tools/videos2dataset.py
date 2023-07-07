import torch
import torch.multiprocessing as mp
#from queue import Queue
import numpy as np
import sys
sys.path.append("../models/")
sys.path.append( "../models/pose_detection/engines/yolov7_pose/")
from pose_detection.engines import YoloV7 as PoseEngine
from hdf5_utils import save_dict_to_hdf5 #save_dict_to_hdf5, load_dict_to_hdf5
import time
import os
import cv2
from datetime import datetime
from threading import Event


def create_folder_tree(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def split_list_into_chunks(lst, chunk_size):
    ''' Splits a list into chunks. Needed for thread processing '''
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
def extract_in_batches(files, buffer, event):
    # -- load model
    model = PoseEngine.Engine()
    j = 0
    for item in files:
        #j += 1
        #print(j)
        total_frames = item[2]
        if item[0].endswith(".avi"):
            print("[Info] Processing: ", item[0])
            # -- open video
            cap = cv2.VideoCapture(item[0])
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in the video
            frame_interval = int(num_frames / total_frames)
            # -- arrays used to save the sequence of a video
            keypoints_sequence = []
            frame_sequence = []
            if frame_interval > 0 :
                for i in range(total_frames):
                    frame_num = i * frame_interval # the exact frame to extract
                    # -- get frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    # -- read frame
                    ret, frame = cap.read()
                    if ret == True: # if frame reading succesful
                        if frame.shape[0] > 365:
                            print(f"frame shape: {frame.shape}")
                            frame = cv2.resize(frame, (640,360  ))
                        keypoints_sequence.append(model.run(frame))
                        frame_sequence.append(frame)
                # -- convert arrays to numpy
                keypoints_sequence = np.stack(keypoints_sequence, axis=0).astype(np.float32)
                frame_sequence = np.stack(frame_sequence, axis=0).astype(np.float32)

                output = {"file_name": item[1], 
                          "img_data": np.expand_dims(frame_sequence, axis=0), 
                          "kp_data": np.expand_dims(keypoints_sequence, axis=0)}

                buffer.put(output)
        # -- free memory
    del model
    torch.cuda.empty_cache()
    event.set()

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


if __name__ == '__main__':
    # -- get videos to work with
    if sys.argv[1] =="unlabeled_videos":
        folder_src = "unlabeled_videos"
    elif sys.argv[1] == "videos2label":
        folder_src = "videos2label"
    else:
        folder_src = None 
        print("Error: arg options are unlabeled_videos or videos2label")

    # --  set mp strategy
    if not torch.backends.mps.is_available():
        mp.set_start_method('fork')
    # -- share items between threads 
    buffer = mp.Queue()
    # -- dataset container for items in queue
    dataset = []
    # -- thread container
    thread_num = 4
    threads = [] 
    events = []
    # -- perform rest of code if folder_src is not empty
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
        print("-- extracting videos --")
        # -- file list
        dir = "./datasets/raw_videos/" + folder_src
        files_fullpath = [[os.path.join(dir, filename), filename, total_frames] for filename in os.listdir(dir)]

        files_in_batches = split_list_into_chunks(files_fullpath, int(len(files_fullpath)//thread_num))[:thread_num]

        # -- start threads
        for i in range(thread_num): # 5 threads
            e = mp.Event()
            events.append(e)
            p = mp.Process(target=extract_in_batches, args=(files_in_batches[i], buffer, events[i], ))
            threads.append(p)

    #start_time = time.perf_counter()
        # -- start threads
        for t in threads:
            t.start()

        event_count = 0

        while True:
            if not buffer.empty():
                dataset.append(buffer.get())
            for e in events:
                if e.is_set():
                    event_count += 1
                    e.clear()
            if event_count == thread_num and buffer.empty():
                print("-- finishing --")
                break

        for t in threads:
            t.join() # blocks process

        # -- save dataset
        props = get_video_props(files_fullpath[0][0], total_frames)
        dataset = {"props": props, "dataset": dataset}
        save_dict_to_hdf5(output_dir, dataset)

