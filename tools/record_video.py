import websockets
import asyncio
import cv2
import base64
import numpy as np 
import sys
from datetime import datetime
import os


def create_folder_tree():
    dirs = ["./datasets/raw_videos/unlabeled_videos","./datasets/raw_videos/used_videos", "./datasets/raw_videos/videos2label"]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

async def getImages(websocket):
    print('Server starting: rpi')
    path = "datasets/raw_videos/unlabeled_videos/"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    save_frames = 30
    out = cv2.VideoWriter(path + 'pose_' + str(round(datetime.now().timestamp())) +'.avi', fourcc, 5.0, (640, 360))
    # -- video count
    video_count = 0
    # -- frame count
    frame_count = 0
    while True:
        image_bytes = await websocket.recv()
        if image_bytes != None:
            frame_count += 1
            # -- decode frame
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            # -- record every 30 frames
            if frame_count % save_frames == 0:
                print(f"Recording video: {video_count} - frame: {frame_count}")
                # -- create new video
                frame_dims = frame.shape
                out = cv2.VideoWriter(path + 'pose_' + str(round(datetime.now().timestamp())) +'.avi', fourcc, 5.0, (frame_dims[1], frame_dims[0]))
                video_count += 1
                # -- reset frame count
                frame_count = 0
            # -- write every other frame
            out.write(frame)

            # -- show frame
            cv2.imshow('image-source: Pi', frame)
            # -- break feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__=='__main__':
    
    PORT = 6000
    HOST = sys.argv[1] #'10.0.0.27'
    
    create_folder_tree()

    async def server():
        async with websockets.serve(getImages,HOST, PORT, ping_interval=10, ping_timeout=None):
            await asyncio.Future()

    asyncio.run(server())
