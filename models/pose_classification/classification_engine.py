import cv2
import sys
import yaml
import time
import torch
import base64
import asyncio
import websockets
import numpy as np
from queue import Queue
from threading import Thread, Event
# -- pose detection model
sys.path.append("../")
sys.path.append("../../train/")
sys.path.append( "../pose_detection/engines/yolov7_pose/")
from pose_detection.engines import YoloV7 as PoseEngine
# -- pose classification model
from pose_classification.AcT import AcT as ClassificationModel
# -- kp normalization
from custom_transforms import kp_norm
from collections import deque
from collections import Counter


 
class Engine():
    def __init__(self):
        # -- set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            #self.device = torch.device('cpu')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') # there is a function in torchvision that is not yet supported: use cpu instead of mps
        else:
            self.device = torch.device('cpu')
        # -- load pose detection
        self.pose_engine = PoseEngine.Engine()
        # -- load pose classification
        self.parameters = self.load_yaml()
        _, self.T, N, C, nhead, num_layer, d_last_mlp, classes = list(self.parameters['MODEL_PARAM'].values())
        batch_size = 1 # process one item at a time
        self.classification_model = ClassificationModel(B=batch_size, 
                                                        T=self.T, # time frames 
                                                        N=N, # number of keypoints
                                                        C=C, # number of channels (x,y, and score)
                                                        nhead=nhead, # number of transformer heads 
                                                        num_layer=num_layer, # number of transformer layers
                                                        d_last_mlp=d_last_mlp, 
                                                        classes=classes,
                                                        device=self.device)
        weights = torch.load("../../weights/model.pth")
        self.classification_model.load_state_dict(weights)
        self.classification_model.to(self.device)
        self.classification_model.eval()
        print(f"[Info] pose classification is in training mode: {self.classification_model.training} running on device: {self.device}")	
        # -- intialize pose buffer
        self.pose_buffer = torch.zeros([self.T,N,C]).to(self.device)
        # -- frame skip
        self.frame_skip = 2
        # -- detection threshold: if classification not confident enough return last confidence class
        self.recognition_history =  deque()
        for i in range(5):
            self.recognition_history.append(None)
        self.label_history = np.array(["none" for i in range(8)])
        self.score_history = np.array([0.01 for i in range(8)])
        # -- classification threshold
        self.class_threshold = 0.90 # during model development set this to 0
        #self.class_threshold = 0. # during model development set this to 0
        # -- set labels
        self.labels = ['sitting', 
                       'standing', 
                       'drinking', 
                       'waving', 
                       'clapping', 
                       'walking', 
                       'picking', 
                       'none']

    # The pose_classification config file: Must be same file the model was trained with
    def load_yaml(self, PATH='../../train/train_config.yaml'):
        stream = open(PATH, 'r')
        dictionary = yaml.safe_load(stream)
        return dictionary 
    
    # pops oldest pose and appends a new one  
    def update_pose_buffer(self, pose):
        pose = torch.tensor(pose)
        current_buff = self.pose_buffer.clone()
        self.pose_buffer[: -1] = current_buff[1: ]
        if torch.backends.mps.is_available():
            self.pose_buffer[-1] = pose.unsqueeze(0).half().to(self.device)
        else:
            self.pose_buffer[-1] = pose.unsqueeze(0).to(self.device)
    
    # -- function to be ran in a seperate threads (works with queues)
    def run_as_thread(self, image_queue, output_queue, event):
        #self.classification_model.eval()
        i = 0
        while True: 
            input_image = image_queue.get() # get image from queue
            time_start = time.perf_counter()
            # -- run pose detection
            #if i % self.frame_skip == 0:
            pose_output, im_wth_kp = self.pose_engine.plot_run(input_image)
            if str(self.device) == "mps":
                pose_output = kp_norm(torch.tensor(pose_output).half().to(self.device))
            else:
                pose_output = kp_norm(torch.tensor(pose_output).to(self.device))
            # -- update queue
            self.update_pose_buffer(pose_output)
            # -- reset counter
            if i == 100:
                i = 0
            # -- run pose classification
            if i % self.frame_skip == 0:
                with torch.no_grad():
                    prediction = torch.exp(self.classification_model(self.pose_buffer.unsqueeze(0)))
                    print(prediction)
                # -- add prediction to queue
                ordered_idx = torch.argsort(prediction, descending=True).tolist()
                ordered_labels = [self.labels[i] for i in ordered_idx]
                ordered_scores = prediction[ordered_idx].cpu().numpy()
                print("O: ", ordered_labels)
                # -- apply threshold
                # -- maybe this logic is better in client side (Java)
                if ordered_scores[0] > self.class_threshold:
                    self.label_history = ordered_labels
                    self.score_history = ordered_scores
                    self.recognition_history.popleft()
                    self.recognition_history.append(ordered_labels[0])
               # else:
               #     prediction[7] = 0.999
               #     ordered_idx = torch.argsort(prediction, descending=True).tolist()
               #     ordered_labels = [self.labels[i] for i in ordered_idx]
               #     ordered_scores = prediction[ordered_idx].cpu().numpy()
               #     self.label_history = ordered_labels
               #     self.score_history = ordered_scores
               #     self.recognition_history.popleft()
               #     self.recognition_history.append(ordered_labels[0])
            # -- respond to queue
            print("History Vote: ", Counter(self.recognition_history).most_common()[0])
            output_queue.put({'labels': self.label_history, 'scores': self.score_history , 'image_with_kp': im_wth_kp, 'history_based_class': Counter(self.recognition_history).most_common()[0]})
            print("Engine Benchmark: ",  time.perf_counter() - time_start)
            print("")
            i += 1
            # -- exit if signal received
            if event.is_set():
                break


# -- Test Zone -- #
if __name__=='__main__':
    # -- intialize pose classification class
    ClassificationEngine = Engine()