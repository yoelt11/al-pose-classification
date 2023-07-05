import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import sys
sys.path.append("./")
from torchvision import transforms
from .utils.datasets import letterbox
from .utils.general import non_max_suppression_kpt
from .utils.plots import output_to_keypoint, plot_skeleton_kpts
import yaml
import time
import os
import urllib.request

class Engine():

    def __init__(self):
        # -- set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            #self.device = torch.device('cpu')
        elif torch.backends.mps.is_available():
            self.device = torch.device('cpu') # there is a function in torchvision that is not yet supported: use cpu instead of mps
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")
        # -- load model
        self.download_model()
        weights = torch.load('/tmp/models/yolov7-w6-pose.pt', map_location=self.device)
        self.model = weights['model']

        if torch.backends.mps.is_available():
            _ = self.model.float().eval()
        else:
            _ = self.model.half().eval()

        self.threshold = 0.55

    def share_memory(self):
        self.model.share_memory()

    
    def download_model(self):
        model_path = "/tmp/models/yolov7-w6-pose.pt"
        dir = "/tmp/models/"

        if not os.path.exists(dir):
            # -- create folder
            print("Directory does not exist -> Creating directory")
            os.makedirs(dir)
            # -- download file
            print("Model does not exist -> downloading file")
            urllib.request.urlretrieve("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt", model_path)
        else:
            if not os.path.exists(model_path):
                print("Model does not exist -> downloading file")
                # -- download file
                urllib.request.urlretrieve("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt", model_path)


    def _preprocessImage(self, source_img):
        # sizes 192, 384, 649, 960
        resized_img =  letterbox(source_img.copy(), 192, stride=64, auto=True)[0]
        resized_img = transforms.ToTensor()(resized_img)
        resized_img = torch.tensor(np.array([resized_img.numpy()]))

        source_img =  letterbox(source_img.copy(), 640, stride=64, auto=True)[0]
        source_img = transforms.ToTensor()(source_img)
        source_img = torch.tensor(np.array([source_img.numpy()]))

        if torch.cuda.is_available(): 
            resized_img = resized_img.half().cuda().to(self.device)
            #resized_img = resized_img.to(torch.float16).to(self.device)
        elif torch.backends.mps.is_available():
            resized_img = resized_img.to(self.device)
        else:
            resized_img = resized_img.half().to(self.device)

        return resized_img, source_img

    def _interpret_output(self, output):
        output = non_max_suppression_kpt(output, 0.05, 0.10, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        return output

    def plot(self, output, image):
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        plot_skeleton_kpts(nimg, output.flatten(), 3)
        return nimg

    def plot_run(self, input_image):
        ''' 
            A function that returns both the keypoints and plotted image.
        '''
        # -- get image from queue
        if input_image.any() != None:
            # -- preprocess input
            resized_img, source_img = self._preprocessImage(input_image)
            # -- run inference
            with torch.no_grad():
                output, _ = self.model(resized_img)
            # -- process output
            output = self._interpret_output(output)
            # -- if output empty return empty array
            if torch.tensor(output).dim() > 1:
                output = output[0, 7:].T.reshape(17,3) # set [k=17,c=3]
            else:
                output = torch.zeros([17,3])
            # -- resize keypoints according to image to plot
            offset_y = -10
                
            inf_h = resized_img.shape[2]
            inf_w = resized_img.shape[3]
            src_h = source_img.shape[2]
            src_w = source_img.shape[3]

            output[:,0] = (output[:,0] / inf_w) * src_w
            output[:,1] = (((output[:,1] / inf_h) * src_h) + offset_y) #* src_h
            # -- plot keypoints to image
            nimg = self.plot(output, source_img)
        return output, nimg

    def plot_thread_run(self, input_queue, response_queue, event):
        ''' 
            A function to ran in a thread. Takes images from an input queue,
            and returns outputs in a response queue.
        '''

        while True:
            # -- get image from queue
            input_image = input_queue.get()
            if input_image.any() != None:
                # -- preprocess input
                resized_img, source_img = self._preprocessImage(input_image)
                # -- run inference
                start_time = time.perf_counter()
                with torch.no_grad():
                    output, _ = self.model(resized_img)
                print(f'Inference time; {time.perf_counter() - start_time}')
                # -- process output
                output = self._interpret_output(output)
                # -- if output empty return empty array
                if torch.tensor(output).dim() > 1:
                    output = output[0, 7:].T.reshape(17,3) # set [k=17,c=3time.perf_counter() - start_time]
                else:
                    output = torch.zeros([17,3])
                # -- resize keypoints according to image to plot
                offset_y = -10
                
                inf_h = resized_img.shape[2]
                inf_w = resized_img.shape[3]
                src_h = source_img.shape[2]
                src_w = source_img.shape[3]

                output[:,0] = ((output[:,0] / inf_w) * src_w) 
                output[:,1] = ((output[:,1] / inf_h) * src_h) + offset_y

                # -- plot keypoints to image
                nimg = self.plot(output, source_img)
                
                # -- send back response
                response_queue.put(nimg)

            # -- break process if signal is set
            if event.is_set():
                break

    def run(self, input_image):
        ''' 
            A function only the keypoint. With identity scaling
            Args:
                input_image: the input image
            Returns:
                output_image: the output image
                keypoints_locs: keypoint x y coordinates
                keypoint_edges: connection between keypoints 
                edge_colors: color for edges 
        '''
        # -- get image from queue
        if input_image.any() != None:
            # -- preprocess input
            resized_img, source_img = self._preprocessImage(input_image)
            # -- run inference
            with torch.no_grad():
                output, _ = self.model(resized_img)
            # -- process output
            output = self._interpret_output(output)
            # -- if output empty return empty array
            if torch.tensor(output).dim() > 1:
                output = output[0, 7:].T.reshape(17,3) # set [k=17,c=3]
            else:
                output = torch.zeros([17,3])
            # -- resize keypoints according to image to plot
            offset_y = -10
                
            inf_h = resized_img.shape[2]
            inf_w = resized_img.shape[3]
            src_h = source_img.shape[2]
            src_w = source_img.shape[3]

            output[:,0] = (output[:,0] / src_w) 
            output[:,1] = (output[:,1] / src_h)  #* src_h) + offset_y) / src_h

        return output

if __name__=="__main__":
    
    #-- load model 
    model = YoloV7()

    # -- input
    #x = np.random.rand(1, 480,640,3).astype(np.float32)
    x = cv2.imread("./models/person.jpg")
    print(x.shape)

    # -- run inference
    #output = model.run(x)
    time_start = time.perf_counter()
    output = model.run(x)
    print(f"Time: {time.perf_counter() - time_start}")
    #plt.imshow(output[1]) 
    #plt.show() 

    print(output.shape)

