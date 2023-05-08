import websockets
import cv2
import base64
import sys
import asyncio
import numpy as np
from queue import Queue
from threading import Thread, Event
from engines import PoseNet as PoseEngine

async def imageServer(websocket):
    while True:
        image_bytes = await websocket.recv()
        if image_bytes != None:
            frame = cv2.imdecode(np.frombuffer(image_bytes, np.unint8), -1)
            image_queue.put(frame)
            # -- get response
            response = response_queue.get()
            cv2.imshow("pose detection", response)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                event.set()
                break


if __name__ == '__main__':
    # -- set port
    PORT = 6000
    # -- set host
    HOST = sys.argv[1]
    # -- image queue
    image_queue = Queue()
    response_queue = Queue()
    event = Event()
    # -- load pose engine
    pose_engine = PoseEngine()

    # -- run image server
    async def image_server():
        async with websockets.serve(imageServer, HOST, PORT):
            await asyncio.Future()

    asyncio.run(image_server())

    # -- inference thread
    inference_thread = Thread(target=pose_engine.plot_run, args=(image_queue, response_queue, event))
    inference_thread.start()
    
    inference_thread.join()
