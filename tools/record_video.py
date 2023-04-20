import websockets
import asyncio
import cv2
import base64
import numpy as np 
import sys



async def getImages(websocket):
    print('Server starting: rpi')
    path = "dataset/raw_videos/"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWwriter(path + 'pose_0.avi', fourcc, 20.0, (640, 480))
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
            # -- record every 25 frames
            if frame_count % 25 == 0:
                print(f"Recording video: {video_count} - frame: {frame_count}")
                out.write(frame)
                video_count += 1
                # -- create new video
                out = cv2.VideoWwriter(path + 'pose_'+ str(video_count) + '.avi', fourcc, 20.0, (640, 480))

            # -- show frame
            cv2.imshow('image-source: Pi', frame)
            # -- break feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__=='__main__':
    
    PORT = 6000
    HOST = sys.argv[1] #'10.0.0.27'

    async def server():
        async with websockets.serve(getImages,HOST, PORT, ping_interval=10, ping_timeout=None):
            await asyncio.Future()

    asyncio.run(server())
