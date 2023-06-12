
import cv2
import sys
import yaml
import base64
import asyncio
import websockets
import numpy as np
from queue import Queue
from threading import Thread, Event
sys.path.append("../")
# -- pose classification model
from classification_engine import Engine as ClassificationEngine

# The image server: sinks image between client and server
async def imageServer(websocket):
    while True:
        image_bytes = await websocket.recv()
        if image_bytes != None:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(image_bytes), np.uint8), -1)
            image_queue.put(frame)
            # -- get response
            response = response_queue.get()
            print("Class Predictions")
            for i in range(len(response['labels'])):
                print(f"{response['labels'][i]}: {str(response['scores'][i])[:5]}")
            print("----------")
            cv2.putText(response['image_with_kp'], 
                        response['labels'][0], (10,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5,
                        (255,42,255), 
                        2)
            cv2.imshow("pose detection", response['image_with_kp'])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                event.set()
                cv2.destroyAllWindows()
                inference_thread.join()
                break

# The pose_classification config file: Must be same file the model was trained with
def load_yaml(PATH='../train/train_config.yaml'):
    stream = open(PATH, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary 

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
    pose_engine = ClassificationEngine()
    # -- inference thread
    inference_thread = Thread(target=pose_engine.run_as_thread,
                               args=(image_queue, response_queue, event,)
                               )
    inference_thread.start()
    # -- initialize image server
    async def image_server():
        async with websockets.serve(imageServer, HOST, PORT, ping_interval=None):
            await asyncio.Future()

    asyncio.run(image_server())
    inference_thread.join()