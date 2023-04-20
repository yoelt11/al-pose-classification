import websockets
import cv2
import asyncio
from queue import Queue
from threading import Thread
import sys
import base64

async def sendImages(address):
    print(address)
    async with websockets.connect(f'ws://{address}:6000') as websocket:
        while True:
            # -- get image from queue
            image = image_queue.get()
            # -- convert to bytes
            image_bytes = cv2.imencode('.jpg', image)[1]
            image_bytes = base64.b64encode(image_bytes)
            # -- send image
            print(f'sending images bytes: {len(image_bytes)}')
            await websocket.send(image_bytes)

def cameraStream():
    video_feed = cv2.VideoCapture(0)
    # -- set video options
    video_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    video_feed.set(cv2.CAP_PROP_FPS, 5)
    
    # -- video loop
    while True:
        ret, frame = video_feed.read(cv2.IMREAD_UNCHANGED)
        image_queue.put(frame)

if __name__ == '__main__':
    image_queue = Queue()
    address = sys.argv[1]
    # -- start camera thread
    camera_thread = Thread(target=cameraStream)
    camera_thread.start()
    # -- send image
    asyncio.run(sendImages(address))

    camera_thread.join()
    
