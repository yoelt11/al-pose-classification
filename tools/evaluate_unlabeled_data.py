import torch
import sys
import glob
import shutil
import os
import numpy as np
import jsonlines as jsonl
sys.path.append("../train/")
#from PoseDatasetUnlabeled import PoseDataset
from PoseDataset import PoseDataset
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import yaml
sys.path.append("../models/")
from pose_classification.AcT import AcT as ClassificationModel
from custom_transforms import kp_norm
from hdf5_utils import load_from_hdf5

def load_yaml(PATH='../train/train_config.yaml'):
    """
        Reads yaml configuration script
    """
    stream = open(PATH, 'r')
    dictionary = yaml.safe_load(stream)
    
    return dictionary 

def loadDataset(batch_size, PATH):
    # -- load your custo65ataset from the .jsonl file
    dataset_props, _, data, filename = load_from_hdf5(PATH)
    # -- load dataset properties
    T = dataset_props['frames_saved'] 
    video_height = dataset_props['height'] 
    video_width = dataset_props['width'] 
    # -- define transformations
    transform = kp_norm
    # -- create instance of dataset
    dataset = PoseDataset(data, filename, transform)
    
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True, num_workers=4, drop_last=True)

    labels = ['sitting', 'standing', 'drinking', 'waving', 'clapping', 'walking', 'picking', 'none']

    return test_loader, labels

def evaluate(loader, model):
    low_conf_vids = []
    video_count = 0
    with torch.no_grad():
        for x, filename in loader:
            network_output = model(x)
            # set y values between 0 and 1
            predictions = network_output.argmax().item()
            max_score = torch.exp(network_output).max().item()
            if max_score < 0.85: # important variable, must increase as model gets more confident
                print(filename)
                low_conf_vids.append(filename[0])
            video_count += 1
    print("overal score: ", (video_count - len(low_conf_vids))/ video_count)
    return low_conf_vids

def update_directories(low_conf_videos):

    # -- Move already labeled videos
    # --- Create directory b if it doesn't exist
    dir_a = "datasets/raw_videos/videos2label/"
    dir_b = "datasets/raw_videos/used_videos/"
    if not os.path.exists(dir_b):
        os.makedirs(dir_b)
    for filename in os.listdir(dir_a):
        # --- Create absolute paths to the file in directory a and destination in directory b
        src = os.path.join(dir_a, filename)
        dst = os.path.join(dir_b, filename)
        # --- Move the file from directory a to directory b
        shutil.move(src, dst)

    # -- add low conf videos
    dir_a = "datasets/raw_videos/unlabeled_videos/"
    dir_b = "datasets/raw_videos/videos2label/"
    if not os.path.exists(dir_b):
        os.makedirs(dir_b)
    for filename in os.listdir(dir_a):
        if filename in low_conf_videos:
            # --- Create absolute paths to the file in directory a and destination in directory b
            src = os.path.join(dir_a, filename)
            dst = os.path.join(dir_b, filename)
            # --- Move the file from directory a to directory b
            shutil.move(src, dst)



if __name__=="__main__":
    file_path = sys.argv[1]
    # -- load parameters
    parameters = load_yaml()
    batch_size, T, N, C, nhead, num_layer, d_last_mlp, classes = list(parameters['MODEL_PARAM'].values())
    batch_size = 1 # process one item at a time
    # -- load dataset
    test_loader, labels = loadDataset(batch_size, PATH="./datasets/unlabeled_datasets/" + file_path)
    # -- load model
    model = ClassificationModel(B=batch_size, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
    if os.path.isfile("../weights/model.pth"):
        print("loading model")
        weights = torch.load("../weights/model.pth")
        model.load_state_dict(weights)
    else:
        print("no pretrained model found in directory")
    # -- evaluate
    low_conf_vids = evaluate(test_loader, model)
    # -- update directories
    update_directories(low_conf_vids)
