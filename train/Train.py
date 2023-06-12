import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import sys
import torch.optim as optim
import numpy as np
from PoseDataset import PoseDataset
from custom_transforms import kp_norm
from torch.utils.tensorboard import SummaryWriter
import os
sys.path.append("../models/")
sys.path.append("../tools/")
from hdf5_utils import load_from_hdf5
from pose_classification.AcT import AcT as ClassificationModel
import yaml
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from datetime import datetime

def load_yaml(PATH='./train_config.yaml'):
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
    # -- split into train / test
    train_proportion = 0.8
    test_proportion = 1 - train_proportion
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(train_proportion * num_samples))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[:split], indices[split:]# define the samplers for train and test data
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, pin_memory=True, num_workers=0, drop_last=True)

    return train_loader, test_loader

def evaluate(loader, model, batch_size, step, writer):
    model.eval() # set model to evaluation mode
    num_correct = torch.tensor([0])
    num_samples = 0
    # -- check if dataset has more than 0
    if len(loader) > 0:
        with torch.no_grad():
            for x, y in loader:
                network_output = model(x)
                # set y values between 0 and 1
                predictions = network_output.argmax(1)
                targets = y.argmax(1)
                num_correct += (predictions == targets).sum()
                num_samples += batch_size
        print("[Test] correct predictions: ", num_correct.item())
        print("[Test] accuracy percentage: ", num_correct.item()/num_samples)

        writer.add_scalar('Test Accuracy:', num_correct.item()/num_samples, global_step=step)
        step += 1
    else: 
        print("[Error] More training samples needed")
    
    return step, writer

def train(loader, model, batch_size, step, writer):
    model.train() # sets model to training mode

    for i, (x, y) in enumerate(loader):

        for param in model.parameters(): # Set grad to zero
            param.grad = None
        network_output = model(x) # predict
        targets = y.max(1)[1]
        # set y values between 0 and 1
        #loss = loss_fn(network_output, y.argmax(1)) # CrossEntropyLoss()
        #loss = loss_fn(network_output, y.argmax(1)) # Nllloss()
        loss = loss_fn(network_output, targets) # Nllloss()
        loss.backward()
        optimizer.step()

        # output and write metrics
        _, predictions = network_output.max(1)
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct)/float(x.shape[0])
        if i % 25 == 0:
            print("[Train] Loss: ", loss.item())

        # update tensorboard
        writer.add_scalar('Training Loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

        step += 1

    print("[Train] Accuracy: ", running_train_acc)
    return step, writer


if __name__=='__main__':
    # -- dataset path
    dataset_path = sys.argv[1]
    # -- load parameters
    parameters = load_yaml()
    batch_size, T, N, C, nhead, num_layer, d_last_mlp, classes = list(parameters['MODEL_PARAM'].values())
    # -- load dataset
    train_loader, test_loader = loadDataset(batch_size, PATH=parameters['DS_PATH'] + dataset_path)
    #-- load model
    model = ClassificationModel(B=batch_size, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
    if os.path.isfile("../weights/model.pth"):
        print("loading model")
        weights = torch.load("../weights/model.pth")
        model.load_state_dict(weights)
    else:
        print("no pretrained model found in directory")
    # -- train parameters
    loss_fn = nn.NLLLoss()
    #loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
            lr=parameters['TRAIN_PARAM']['LEARNING_RATE'],
            weight_decay=parameters['TRAIN_PARAM']['WEIGHT_DECAY'])
    num_epochs = parameters['TRAIN_PARAM']['EPOCHS']
    # -- tensorboard
    writer = SummaryWriter(parameters['TB_PATH']+'/'+ str(round(datetime.now().timestamp())))
    # --  train log
    train_step = 0
    test_step = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch}')
        # -- train step
        train_step, writer = train(train_loader, model, batch_size, train_step, writer)
        # -- test step
        test_step, writer = evaluate(test_loader, model, batch_size, test_step, writer)
        # -- save every 20 epochs
        if epoch % 20 == 0:
            torch.save(model.state_dict(), parameters['MODEL_OUT_PATH'])
    
    torch.save(model.state_dict(), parameters['MODEL_OUT_PATH'])



