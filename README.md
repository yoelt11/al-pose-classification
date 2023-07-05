# A framework for the development of pose classification models

___

## Index

1. Introduction

2. Pose detection engine

3. Pose classification engine

4. Framework modules
   
   1. Record videos
   
   2. Extract videos
   
   3. Label dataset
   
   4. Merge dataset
   
   5. Train
   
   6. Evaluate labeled dataset
   
   7. Evaluate unlabeled dataset
   
   8. Live preview

5. Annex
   
   1. Dependencies
   
   2. How to use

---

## 1. Introduction

This project is aimed to define a systematic approach for training of pose classification models. It packages all the important tools for the creation and evaluation as well as a deployment examples. The framework is depicted in figure 1. 

The workflow of the framework is as follows:

1) First videos are recorded with the  "tools/record_videos.py"

2) The videos are extracted (converted into .h5 format) with the  "tools/videos2dataset.py"

3) The videos are labeled with the "tools/label_dataset.py"

4) Here the possibility to merge "old_labeled_dataset"  with a new "labeled_dataset" is available with the "merge_datasets.py", this is useful in case the resulting trained model is not accurate and more training data is needed.

5) The training is performed based on either a "merged_dataset" or a "labeled_dataset", where the "merged_dataset" is a combination of multiple "labeled_datasets".

6) Finally the model is evaluated based on the method selected by the user:
   
   1) Live Preview: Tests the model with a video feed, here one can detect which poses are not being accurately classified such that more samples can be recorded with step 1.
   
   2) Evaluate Labeled Dataset:  Runs the model on the train data and prints a graph of the accuracy of each pose
   
   3) Evaluate Unlabeled Dataset: Runs the model on the unlabeled videos and seperates the ones with low accuracy such that they can be used for training.

![image info](/Users/developer/Documents/computer-vision-dev/al-pose-classification/figures/active-learning-pose-class-workflow.png)

Fig 1. The framework workflow.

The different blocks of the framework depicted in figure 1 interact with the file system according to figure 2. For instance, **Record video** produces "*.avi" files and stores them in the "tools/Datasets/raw_video" folder, then the **Extract Video** takes the videos from the "tools/Datasets/raw_video" and creates a "*.h5" dataset in the unlabeled dataset, and so on...  

Zeit![image info](/Users/developer/Documents/computer-vision-dev/al-pose-classification/figures/framework_file_interaction.drawio.png)

Fig 2. The interaction between different modules and the file system. Inputs and outputs are shown in blue and purple colors respectively. Red arrows denote a bidirectional relation.

Each of the modules shown in figures 1 and 2 are further explained in chapter 4. For a quick start go to Chapter 5.2 "How to use".

---

## 2. Pose Detection Engine

The pose detection engine is aimed to provide the keypoint coordinates of a person in a given frame. There are several models that have been developed for this purpose. This framework provides thre different models, namely, movenet (coral), posenet (pytorch) and yolov7-pose (pytorch).  This last model is used by default since it is the most accurate and offers less latency than the other models. These engines are located in:

```bash
models/pose_detection/engines
```

The afore models given an input image consiting of dimensions [H, W, C=3], produces an output consisting of the keypoint coordinates and scores of dimensions [K=17, C=3], where K is the keypoints, and C=0 is the X coordinate, C=1 is the Y coordinate, and C=2 is the score. The keypoint numbering is shown in figure 3. The outputs are the normalized coordiantes w.r.t. the original images dimension, therefore, to resize these keypoints to any image of H and W, the coordinates need to be multiplied by the corresponding dimension (H or W).

```python
   output[:,0] = (output[:,0] / src_w) 
   output[:,1] = (output[:,1] / src_h) 
```

<img title="" src="file:///Users/developer/Documents/computer-vision-dev/al-pose-classification/figures/pose_keypoints.drawio.png" alt="image info" data-align="center">

                                            Fig.3 Pose detection keypoint numbering.

The pose detection engines provides three infrence modes: 

1) Run as Thread (plot_run_thread)

2) Plot and run (plot_run)

3) Simple Run (run) 

#### Run as Thread

This run mode that is meant to run as a thread, and comunicates via queues to external components:

```python
    from engines import YoloV7 as PoseEngine

    pose_engine = PoseEngine.Engine() # the pose engine class
    # -- the endpoints
    image_queue = Queue() 
    response_queue = Queue()
    # -- thread initialization
    inference_thread = Thread(target=pose_engine.plot_thread_run, args=(image_queue, response_queue, event))
    inference_thread.start()
```

As shown in the example above the PoseEngine.Engine() class contains a method called plot_thread_run, which takes as inputs an image_queue, responsible of supplying the image, and a response_queue, queue in which the engine writes the results from the detection.

#### Plot and Run

This mode is a function that takes an image as an input, and returns the pose_output and the image with the keypoints plotted. Below is an example fo how this mode can be used .

```python
from engines import YoloV7 as PoseEngine

pose_engine = PoseEngine.Engine()

pose_output, plotted_image = pose_engine.plot_run(input_image)

# here input_image is an image np.array or torch.array of shape [H,W,C=3]
# the image is resized according to model in the plot_run function
```

### Run

This mode is the same as the previous one with the only difference that it only returns the pose_output.

An example of a use case of this engine is available in the following directory:

```
models/pose_detections/main_pose.py
```

This script assumes that a stream of image is bein sent via port 6000 and uses the run as thread mode. 

---

## 3. Pose Classification Engine

This engine aims to provide the model to be developed by the framework, model which classifies a sequence of pose_outputs provided by the pose detection engine. The model is based on [Action Transormer]([[2107.00606] Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition](https://arxiv.org/abs/2107.00606)).  The classification engine serves as a wrapper over the model which makes the execution and testing of the model easy.

The pose classification is meant to be ran as a thread using the run_as_thread method provided by the engine.  See code below as example on how to run.

```python
from classification_engine import Engine as ClassificationEngine
from queue import Queue
from threading import Thread, Event


image_queue = Queue()
response_queue = Queue()
event = Event()

pose_engine = ClassificationEngine()

inference_thread = Thread(target=pose_engine.run_as_thread,
                               args=(image_queue, response_queue, event,)
                               )
inference_thread.start()
```

---

## 4. Framework Modules

### 4.1 Record Videos

```bash
python3 record_videos.py
```

This tools can be executed as shown below, and it utilizes a websocket image stream that can be executed, e.g. from a raspberry pi or other image source. This is a script that continuosly saves a video every 30 frames. The idea of this script is to capture the poses in a more natural way and then classify each of the videos.

The default output directory is as follows:

```python
/tools/datasets/raw_videos/unlabeled_videos/
```

From here the videos can be manually moved to the videos2label folder to mantain the files more organized, since you might want to only label a subset of all the videos captured.

The following script can be run in the camera device that is to provide images via the websocket connection:

```python
# -- client_video_src.py (source) e.g. rpi
# -- ran as "python3 client_video_src.py 10.0.0.83" where the
# -- ip address should be modified according to the device running
# -- the record_video.py

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
```

### 4.2 Extract Videos

```bash
python3 videos2dataset.py source # source either videos2label or unlabeled_videos
```

This script can be executed as shown above, it takes as argument the folder source cotanining the videos to be converted into a database. In other words, the videos should be locaded in the following directories:

```
/tools/datasets/raw_videos/videos2label
or
/tools/datasets/raw_videos/unlabeled_videos
```

The structure of the dataset is as follows:

```json
{
    "props": {
        "frame_count": 30,
        "fps": 5,
        "frames_saved": 20,
        "width": 640,
        "height": 360 
    },
    "dataset": [
        {
            "file_name": "pose_*.avi",
            "img_data": img_data,
            "kp_data": kp_data
        }
    ]
}
```

The output folder is the following:

```
/tools/datasets/unlabeled_datasets/{file_name}_*.h5
```

Where the file name can be unlabeled_dataset*.h5 or videos2label*.h5. The reason why there is distinction is because the "evaluate_unlabeled_data.py" uses the unlabeled_dataset*.h5 to sort the videos in different folders according to the defined confidence score. While the "label_dataset.py" uses the videos2label*.h5 dataset to label the dataset.

### 4.3 Label Dataset

```bash
python3 label_dataset.py <dataset_name>
```

Datasets can be labeled as shown above, here the <dataset_name> is the file name of the dataset. This file should be located in the /tools/datasets/unlabeled_dataset directory.

This script first plays a video, and then waits for an input with the corresponding label.

The label options are: drinking, sitting, waving, clapping, picking, standing, none. To replay the video, press enter "or input an empty string". If the video does not belong to any class or  is "tricky" to determine the correct class you can enter the word "skip" to skip this video in the final dataset.

The output dataset is as follows:

```json
{
    "props": {
        "frame_count": 30,
        "fps": 5,
        "frames_saved": 20,
        "width": 640,
        "height": 360 
    },
    "dataset": [
        {
            "file_name": "pose_*.avi",
            "img_data": img_data,
            "kp_data": kp_data,
            "target": [0,0,0,0,0,0,0,1],
            ""
        }
    ]
}
```

This resulting dataset can be directly used for training a model.

### 4.4 Merge Dataset

```bash
python3 merge_h5_datasets.py <dataset_1> <dataset_2> ... <dataset_n>
```

This script merges two or more datasets together. The datasets should be located in the folder "/tools/dataset/labeled_datasets".  

As part of an "active learning scheme", sometimes it is desired to combine the datasets from different iterations in the stage of learning, for this reason this script is created.

**Note:** Because the original labeled datasets contain the full videos, merging datasets might result in really big memory intensive datasets, for this reason, the datasets are saved without the img_data. 

The resulting dataset can be use directly for training the model .

### 4.5 Train

```bash
python3 Train.py <dataset_file>
```

This script is used to train the model. Here the <dataset_file> corresponds to the labeled_dataset located in the "labeled_dataset" folder. The output model is saved in the "/weights" directory. If a file is already located in this folder then the model is loaded again and the training is resumed from where it left off. 

This script uses a train_config.yaml file located in the "/train/" directory. This files contains the basic model configuration as well as the training configuration.

```yaml
# ./train/train_config.yaml
#---- MODEL PARAMETERS
MODEL_PARAM: # input in the same order as class initialization
  BATCH_SIZE: 128
  TIME_FRAMES: 20
  KEYPOINTS: 17
  CHANNEL_IN: 3
  HEADS: 8
  LAYERS: 2
  LAST_MLP: 512
  CLASSES: 8

#---- MODEL CONFIG
DS_PATH: '../tools/datasets/labeled_datasets/'
TB_PATH: '../tools/logs/'
TB_COMMENT: '_our_ds'
MODEL_OUT_PATH: '../weights/model.pth'

#---- TRAINING PARAMETERS
TRAIN_PARAM:
  LEARNING_RATE: .00025
  WEIGHT_DECAY: .0001
  EPOCHS: 500
```

The default configuration is shown above. 

### 4.6 Evaluate Labeled Dataset

```bash
python3 evalualte_labeled_dataset.py <labeled_dataset.h5>
```

This script as the name tells, evaluates the model on a labeled_dataset. The output of this script is a bar graph containing the accuracy per pose class.

![image](/Users/developer/Documents/computer-vision-dev/al-pose-classification/figures/plot_2.png)

### 4.7 Evaluate Unlabeled Dataset

```bash
python3 evaluate_unlabeled_dataset <unlabeled_dataset.h5>
```

This script is meant to seperate unsued videos into videos with low confidence and videos with high confidence. High confidence videos are not needed, however, videos with low confidence can be used to refine the network in areas where the model is lacking accuracy, thus, serving as fundamental part of an active learning scheme.

The script uses an unlabeled_dataset.h5 located in the "/tools/datasets/unlabeled_dataset" path. The selected dataset must match the videos located in the "/tools/datasets/raw_videos/unlabeled_videos". The model is then ran on the unlabeled dataset and moves the low accuracy videos to the "/tools/datasets/raw_videos/videos2label".

After the low accuracy videos are moved to the "/tools/datasets/raw_videos/videos2label". Then those videos can be extracted, labeled and merged with starting dataset. After the training process can be executed again.

### 4.8 Live Preview

```bash
/models/pose_classification/module_test.py 0.0.0.0
```

This is the best way to test a trained model. This script ask as a websocket server that receive an image feed from a websocket client. In order for this script to work a websocket client that streams the images as mentioned in section 4.1 needs to be running.

---

## 5. Annex

### 5.1 Dependencies

The dependencies for this project are listed in the requirements.txt

```python
opencv-python==4.7.0.68
websockets==10.4
torch==2.0.0
jsonlines==3.1.0
matplotlib==3.7.1
tensorboard==2.13.0
PyYAML==6.0
tqdm==4.65.0
torchvision=0.15.2
pandas==2.0.1
seaborn==0.12.2
scipy==1.10.1
h5py==3.8.0
```

## 5.2 How to use

In order to use this framework one must first install the dependencies. To do this it is best to create a python environment. To do this <u>you must have pyenv installed along with the virtualenv plugin</u>. 

In virtual env install new python version

```bash
pyenv install 3.10-dev
```

Create new virutalenv

```bash
pyenv virtualenv 3.10-dev "pose-class-framework"
```

Start virtualenv

```bash
pyenv activate pose-class-framework
```

Install requirements, go to project's root directory and execute

```bash
pip install -r requirements.txt
```

Once this is done you should be able to run any of the scripts in this project.

## Use case Example

0. Install dependencies

1. Activate the video source stream
   
   ```bash
   python3 client_video_src.py 10.0.0.83 # replace ip with framework ip
   ```

2. Record videos
   
   ```bash
   python3 record_videos.py
   ```
   
   videos should appear in /tools/datasets/raw_videos/unlabeled_videos/. From here move the videos to /tools/datasets/raw_videos/videos2label

3. Extract dataset from recorded videos
   
   ```bash
   python3 videos2dataset.py videos2label
   ```
   
   The output dataset should be located in /tools/datasets/unlabeled_dataset/ . And the should be named videos2label_*.h5

4. Label dataset
   
   ```bash
   python3 label_dataset.py videos2label_*.h5 
   ```
   
   The asterisc should be replaced according to the dataset created in previous step. The output file is located in /datasets/labeled_datasets, and should be named labeled_dataset_*.h5, 

5. Train model
   
   ```bash
   Train.py labeled_dataset_*.h5
   ```
   
   Asterics should be replaced according to file dataset created in previous step.  The output file of this step is located in "/weights/"

6. Evaluate model

7. Repeat Steps

---
