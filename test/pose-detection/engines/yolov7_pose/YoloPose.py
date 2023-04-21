import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

class YoloPose():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load("weights/yolov7-w6-pose.pt", map_location=self.device)
        self.model = weights['model'].to(self.device).eval()

    def preprocess_image(self, image):
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        return image

    def plot_image(self, image, inference_out):
        pass

    #def plot_run(self, image_queue, response_queue, event):
    def plot_run(self, image):
        #while True:
        # -- close thread if signal received
        #if event.is_set():
        #    break
        # -- get image
        #image = image_queue.get()
        # -- preprocess image according to model input
        x = self.preprocess_image(image)
        # -- run inference
        y, _ = self.model(x)
        # -- plot image
        #image_out = self.plot_image(image, y)
        # -- send response
        #response_queue.put(image_out)
        return y

            

    def run():
        pass

#=  Test Zone =#
if __name__=="__main__":

    # -- initialize model
    engine = YoloPose()

    # -- create dummy input
    image = torch.rand([640,360,3], dtype=torch.half).numpy()

    # -- run moodel
    print(engine.plot_run(image).shape)

