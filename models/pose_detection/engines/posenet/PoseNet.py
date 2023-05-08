import cv2
import torch
from . import posenet
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('tkagg')

class PoseNet():

    def __init__(self):
        self.interpreter = posenet.load_model(101)
        self.output_stride = self.interpreter.output_stride
        self.scale_factor = 0.7125
        self.threshold = 0.55

    def valid_resolution(self, width, height, output_stride=16):
        target_width = (int(width) // output_stride) * output_stride + 1
        target_height = (int(height) // output_stride) * output_stride + 1
        return target_width, target_height

    def _preprocessImage(self, source_img, scale_factor=1.0, output_stride=16):
        """Returns Resized image according to model
        Args:
            image: input image
        Returns:
            * Resized Image
            """
        target_width, target_height = self.valid_resolution(
            source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
        scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

        input_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        input_img = cv2.resize(input_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        input_img = input_img * (2.0 / 255.0) - 1.0
        input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)

        return input_img, source_img, scale
        
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
            
            response_queue.put(ref_image)

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

        input_image, display_image, output_scale = self._preprocessImage(input_image)
        input_image = torch.from_numpy(input_image)

        with torch.no_grad():

             #mod_image = torch.unsqueeze(torch.tensor(input_image),0).permute(0,3,1,2).float()

             heatmaps, offset, displacement_fwd, displacement_bwd = self.interpreter(input_image.float())
             pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                 heatmaps.squeeze(0),
                 offset.squeeze(0),
                 displacement_fwd.squeeze(0),
                 displacement_bwd.squeeze(0),
                 output_stride=self.output_stride,
                 max_pose_detections=1,
                 min_pose_score=0.15
             )

        output_tensor = np.zeros([1,1,17,3])
        output_tensor[:,:,:,0] = keypoint_coords[:,:,0]# * 360 # y
        output_tensor[:,:,:,1] = keypoint_coords[:,:,1]# * 640 # x
        output_tensor[:,:,:,2] = keypoint_scores

        return output_tensor.squeeze()

if __name__=="__main__":
    
    #-- load model 
    model = PoseNet()

    # -- input
    #x = np.random.rand(1, 480,640,3).astype(np.float32)
    x = cv2.imread("test.png")
    print(x.shape)

    # -- run inference
    #output = model.run(x)
    output = model.plot_run(x)
    plt.imshow(output[1]) 
    plt.show() 

    print(output[0].shape)

