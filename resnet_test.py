import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
from Resnet.resenet_model import Net
from preprocess import Preprocess


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd

# from __future__ import print_function
import torch  # pip3 install torch torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchsummary import summary  # pip3 install torch-summary

from torchvision import datasets, transforms

import os
import re
import pickle
import time
from tqdm import tqdm
import sys
from PIL import Image

d = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, '18': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18}
m = {str(value):int(key) for key, value in d.items()}

def get_predicted_label(img, device, model):  # numpy array get from the previous
    with torch.no_grad():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, "RGB")
        data_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
        ])
        data = data_transform(img)
        data = data.unsqueeze(0)
        data = data.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        r = m[str(pred.item())]
        return r


def main():
    correct = 0
    total = 0
    for i in range(19):
        v_list = pickle.load(open("generated/V_list", "rb"))
        v_list = v_list[-1]
        condition_list = pickle.load(open("generated/condition_list", "rb"))
        condition_list = condition_list[-1]
        epsilons = 0
        use_cuda = True
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        model = Net(l=512)
        model = model.to(device)
        model.load_state_dict(torch.load('/home/lu677/cs490/cs490-research/Resnet/epochs/model_150.pth'))
        model.eval()
        folder = "./Test/" + str(i)
        g = os.walk(folder)
        dict_list = []
        for path, dir_list, file_list in g:
            for file_name in tqdm(file_list, desc=folder):
                path_file = os.path.join(path, file_name)
                original_label = path.split("/")[-1]
                x_new = Preprocess.preprocess_image_gaussian(path_file, v_list, epsilons)
                output_label_x_new = get_predicted_label(x_new, device, model)
                if output_label_x_new == i:
                    correct += 1
                total += 1
    print("Accuracy: ", correct / total)


main()


def final():

    train_dir = "/home/lu677/cs490/cs490-research/TrainVal/Train"
    valid_dir = "/home/lu677/cs490/cs490-research/TrainVal/Val"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    l_epochs = [100, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    layer = 512
    l_layer = [64, 128, 256, 512, 1024, 2048, 4096]
    batch_size = 16
    l_batch_size = [2, 8, 16, 32, 64, 128]
    learning_rate = 0.01
    l_learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    momentum = 0.9
    l_momentum = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    weight_decay = 1e-6
    l_weight_decay = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    nesterov = False
    l_nesterov = [True, False]
    log_interval = 10
    epochs = 151
    load_from = "/home/lu677/cs490/cs490-research/Resnet/epochs/model_150.pth"
    first_epoch = 100
    save_to = "/home/lu677/cs490/cs490-research/Resnet/epochs/"

    l_train_loss = []
    l_val_loss = []

    data_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3394, 0.3081, 0.3161), (0.2753, 0.2631, 0.2685))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transform)
    print(train_dataset.class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_dataset = datasets.ImageFolder("/home/lu677/cs490/cs490-research/Test", transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = Net(l=layer).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if load_from != '':
        model.load_state_dict(torch.load(load_from))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay, nesterov=nesterov)
    train_loss = None
    val_loss = None
    validation(model, test_loader, device)


def validation(model, valid_loader, device):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            if correct == 0:
                pickle.dump(data, open("data", "wb"))
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).data  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(valid_loader.dataset)
    print('\nValidation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        validation_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

    return validation_loss.detach().cpu().numpy()


# final()
