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

class YoloV7():

    def __init__(self):
        # -- set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")
        # -- load model
        print(os.getcwd())
        weights = torch.load('./engines/yolov7_pose/models/yolov7-w6-pose.pt', map_location=self.device)
        self.model = weights['model']
        _ = self.model.float().eval()

        self.threshold = 0.55

    def _preprocessImage(self, source_img):
        # sizes 192, 384, 649, 960
        resized_img =  letterbox(source_img.copy(), 192, stride=64, auto=True)[0]
        resized_img = transforms.ToTensor()(resized_img)
        resized_img = torch.tensor(np.array([resized_img.numpy()]))

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            resized_img.half().to(self.device)

        return resized_img, source_img

    def _interpret_output(self, output):
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        return output

    def plot(self, output, image):
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        return nimg


       
    def plot_run(self, input_image):
        # -- save original copy
        ref_image = input_image.copy()
        # -- run inference
        output = self.run(input_image)
        # -- plot to image
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for node in range(output.shape[2]):
                    if output[i,j,node,2] > self.threshold:
                        kp_x = output[i,j,node,1]
                        kp_y = output[i,j,node,0]
                        ref_image = cv2.circle(ref_image, (int(kp_x), int(kp_y)), radius=5, color=(255,255,0))

        return output, ref_image

    def plot_thread_run(self, input_queue, response_queue, event):
        while True:
            print("--")
            input_image = input_queue.get()
            if input_image.any() != None:
                s_t = time.perf_counter()
                resized_img, source_img = self._preprocessImage(input_image)
                print(f"image resizing: {time.perf_counter() - s_t}")

                s_t = time.perf_counter()
                with torch.no_grad():
                    output, _ = self.model(resized_img)
                print(f"model inference: {time.perf_counter() - s_t}")

                s_t = time.perf_counter()
                output = self._interpret_output(output)
                print(f"model interpretation: {time.perf_counter() - s_t}")

                s_t = time.perf_counter()
                nimg = self.plot(output, resized_img)
                print(f"image plotting: {time.perf_counter() - s_t}")

                response_queue.put(nimg)

            if event.is_set():
                break

    def run(self, input_image):
        """Runs model and returns interpreted network outputs
        Args:
            input_image: the input image
        Returns:
                output_image: the output image
                keypoints_locs: keypoint x y coordinates
                keypoint_edges: connection between keypoints "
                edge_colors: color for edges """
        #print(torch.tensor(input_image).unsqueeze(0).numpy().shape)

        resized_img, source_img = self._preprocessImage(input_image)

        with torch.no_grad():
            output, _ = self.model(resized_img)
        output = self._interpret_output(output)

        nimg = self.plot(output, resized_img)
        #cv2.imwrite("./new_img.jpg", nimg)

        return output, nimg

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

