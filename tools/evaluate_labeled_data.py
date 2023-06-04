import torch
import sys
import glob
import os
import numpy as np
sys.path.append("../train/")
from PoseDataset import PoseDataset
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import yaml
sys.path.append("../models/")
from pose_classification.AcT import AcT as ClassificationModel
from hdf5_utils import load_from_hdf5
from custom_transforms import kp_norm

def load_yaml(PATH='../train/train_config.yaml'):
    """
        Reads yaml configuration script
    """
    stream = open(PATH, 'r')
    dictionary = yaml.safe_load(stream)
    
    return dictionary 

def loadDataset(batch_size, PATH):
    # -- load your custom dataset from the .jsonl file
    dataset_props, _, data, _, targets = load_from_hdf5(PATH)
    # -- load dataset properties
    T = dataset_props['frames_saved'] 
    video_height = dataset_props['height'] 
    video_width = dataset_props['width'] 
    # -- define transformations
    transform = kp_norm
    # -- create instance of dataset
    dataset = PoseDataset(data, targets, transform)
    
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True, num_workers=4, drop_last=True)

    labels = ['sitting', 'standing', 'drinking', 'waving', 'clapping', 'walking', 'picking', 'none']

    return test_loader, labels

def evaluate(loader, model):
    correct = torch.zeros(8)
    total = torch.zeros(8)
    with torch.no_grad():
        for x, y in loader:
            network_output = model(x)
            # set y values between 0 and 1
            predictions = network_output.argmax().item()
            print(predictions)
            targets = y.argmax().item()
            print(f"t: {targets}, p: {predictions}")
            total[targets] +=1
            if predictions == targets:
                correct[predictions] += 1
    metrics  = correct / total 
    metrics = torch.where(torch.isnan(metrics), torch.zeros_like(metrics), metrics)
    return metrics

def plot_and_save(metrics, labels):
    save_dir = "./metrics"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plot_files = glob.glob(os.path.join(save_dir, 'plot_*.png'))
    
    # Get the highest plot number
    if len(plot_files) > 0:
        last_plot_number = max([int(os.path.basename(f).split('.')[0].split('_')[1]) for f in plot_files])
    else:
        last_plot_number = 0

    # Increment the plot number
    plot_number = last_plot_number + 1
    # -- plot image
    plt.figure(figsize=(10, 6))
    plt.xticks([i for i in range(len(labels))], labels, rotation=45)
    plt.bar([i for i in range(len(labels))],metrics.numpy())
    plt.savefig(os.path.join(save_dir, f'plot_{plot_number}.png'))

if __name__=="__main__":
    file_path = sys.argv[1]
    # -- load parameters
    parameters = load_yaml()
    batch_size, T, N, C, nhead, num_layer, d_last_mlp, classes = list(parameters['MODEL_PARAM'].values())
    batch_size = 1 # process one item at a time
    # -- load dataset
    test_loader, labels = loadDataset(batch_size, PATH=parameters['DS_PATH'] + file_path)
    # -- load model
    model = ClassificationModel(B=batch_size, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
    if os.path.isfile("../weights/model.pth"):
        print("loading model")
        weights = torch.load("../weights/model.pth")
        model.load_state_dict(weights)
        model.eval()
    else:
        print("no pretrained model found in directory")
    # -- evaluate
    metrics = evaluate(test_loader, model)
    print(metrics)
    # -- plot
    plot_and_save(metrics, labels)
