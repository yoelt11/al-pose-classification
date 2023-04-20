import websockets
import asyncio
import cv2
import base64
import numpy as np 
import sys



async def getImages(websocket):
    print('Server starting: rpi')
    while True:
        image_bytes = await websocket.recv()
        if image_bytes != None:
            # -- decode frame
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
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
