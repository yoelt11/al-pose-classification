import cv2
import numpy as np
import sys
sys.path.append("./")
import yaml
import time
import os
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
import urllib.request

class Engine():

    def __init__(self):
        # -- set device
        
        model_path = "/tmp/models/movenet_single_pose_thunder_ptq_edgetpu.tflite"
        dir = "/tmp/models/"

        if not os.path.exists(dir):
            # -- create folder
            os.makedirs(dir)
            # -- download file
            print("downloading file")
            urllib.request.urlretrieve("https://raw.githubusercontent.com/google-coral/test_data/master/movenet_single_pose_thunder_ptq_edgetpu.tflite", model_path)
        else:
            if not os.path.exists(model_path):
                print("downloading file")
                # -- download file
                urllib.request.urlretrieve("https://raw.githubusercontent.com/google-coral/test_data/master/movenet_single_pose_thunder_ptq_edgetpu.tflite", model_path)

        self.interpreter = make_interpreter("/tmp/models/movenet_single_pose_thunder_ptq_edgetpu.tflite")
        self.interpreter.allocate_tensors()
        self.threshold = 0.35

    def _preprocessImage(self, source_img):
                # -- thunder h=256, w=256
        resized_img = source_img.copy()
        resized_img = cv2.resize(resized_img, common.input_size(self.interpreter), interpolation = cv2.INTER_AREA)
        return resized_img, source_img

    def plot(self, output, source_image):
        for node in range(output.shape[0]):
            if output[node,2] > self.threshold:
                kp_x = output[node,1]
                kp_y = output[node,0]
                source_image = cv2.circle(source_image, (int(kp_x), int(kp_y)), radius=5, color=(255,255,0), thickness=-1)
        return source_image

    def plot_run(self, input_image):
        ''' 
            A function that returns both the keypoints and plotted image.
        '''
        print("--")
        # -- get image from queue
        if input_image.any() != None:
            resized_img, source_img = self._preprocessImage(input_image)
            # -- run inference
            common.set_input(self.interpreter, resized_img)
            self.interpreter.invoke()
            output = common.output_tensor(self.interpreter, 0).copy().reshape(17, 3)
            # -- resize keypoints according to image to plot
            inf_w = 1 # resized_img.shape[2]
            inf_h = 1 # resized_img.shape[3]
            src_w = source_img.shape[1]
            src_h = source_img.shape[0]

            output[:,0] = (output[:,0] / inf_h) * src_h
            output[:,1] = (output[:,1] / inf_w) * src_w
        return output, source_img

    def plot_thread_run(self, input_queue, response_queue, event):
        ''' 
            A function to ran in a thread. Takes images from an input queue,
            and returns outputs in a response queue.
        '''

        while True:
            print("--")
            # -- get image from queue
            input_image = input_queue.get()
            if input_image.any() != None:
                # -- preprocess input
                resized_img, source_img = self._preprocessImage(input_image)
                # -- run inference
                start_time = time.perf_counter()
                common.set_input(self.interpreter, resized_img)
                self.interpreter.invoke()
                output = common.output_tensor(self.interpreter, 0).copy().reshape(17, 3)
                print(f'Inference time; {time.perf_counter() - start_time}')
                # -- resize keypoints according to image to plot
                inf_w = 1 # resized_img.shape[2]
                inf_h = 1 # resized_img.shape[3]
                src_w = source_img.shape[1]
                src_h = source_img.shape[0]

                output[:,0] = (output[:,0] / inf_h) * src_h
                output[:,1] = (output[:,1] / inf_w) * src_w

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
                keypoint_edges: connection between keypoints# 
                edge_colors: color for edges 
        '''
        print("--")
        # -- get image from queue
        if input_image.any() != None:
            # -- preprocess input
            resized_img, source_img = self._preprocessImage(input_image)
            # -- run inference
            common.set_input(self.interpreter, resized_img)
            self.interpreter.invoke()
            output = common.output_tensor(self.interpreter, 0).copy().reshape(17, 3)
        return output

if __name__=="__main__":
    
    #-- load model 
    model = Engine()

    # -- input
    #x = np.random.rand(1, 480,640,3).astype(np.float32)
    x = cv2.imread("./person.jpg")
    print(x.shape)

    # -- run inference
    #output = model.run(x)
    while True:
        time_start = time.perf_counter()
        output = model.run(x)
        print(f"Time: {time.perf_counter() - time_start}")
    #plt.imshow(output[1]) 
    #plt.show() 

    print(output.shape)

