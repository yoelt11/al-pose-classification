import torch

class YoloPose():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load("weights/yolov7-w6-pose.pt")
        self.model = weights['model'].to(self.device).eval()

    def preprocess_image(self, image):
        continue

    def plot_image(self, image, inference_out):
        continue

    def plot_run(self, image_queue, response_queue, event):
        while True:
            # -- close thread if signal received
            if event.is_set():
                break
            # -- get image
            image = image_queue.get()
            # -- preprocess image according to model input
            x = self.preprocess_image(image)
            # -- run inference
            y = self.model(x)
            # -- plot image
            image_out = self.plot_image(image, y)
            # -- send response
            response_queue.put(image_out)

            

    def run():
        pass
