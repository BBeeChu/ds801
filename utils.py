import os
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import math
import torch

def list_files_in_directory(directory_path):
    try:
        # 디렉토리 내의 모든 파일과 디렉토리를 가져옵니다.
        files_and_directories = os.listdir(directory_path)
        
        # 파일만 필터링합니다.
        files = [f for f in files_and_directories if os.path.isfile(os.path.join(directory_path, f))]
        
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []
    
class ImageDataset(Dataset):
    def __init__(self, base_path, image_dir, labels, transform):
        self.base_path = base_path
        self.image_dirs = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        raw_image = Image.open(os.path.join(self.base_path, image_dir))
        image = self.transform(raw_image)
        label = self.labels[idx]
        return image, label
    



def train(net, train_dataset, train_loader, valid_dataset, valid_loader, device, criterion, optimizer, scheduler, args):
    train_errors, validation_errors = [], []
    total_epochs = args.total_epochs
    # early_ratio = args.early_ratio
    # early_stop_epochs = math.ceil(total_epochs*early_ratio)
    # Training for 30 epochs
    for epoch in range(total_epochs):  # loop over the dataset multiple times
    # an epoch indicates the time to update the model for all training data points.

        # [CODE for TRAINING A SINGLE EPOCH]
        net.train()

        running_loss = 0.0
        num_iter = len(train_loader)
        correct = 0 # the number of correct cases
        for i, data in enumerate(tqdm(train_loader), 0):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, true = torch.max(labels.data, 1)
            
            correct += (predicted == true).sum().item()

            # print statistics
            running_loss += loss.item()

        print('[%d epoch] train loss: %.3f, train acc: %.3f' %
            (epoch + 1, running_loss / num_iter, correct / len(train_dataset)))
        train_errors.append(1.0 - correct / len(train_dataset))
        running_loss = 0.0
        scheduler.step()

        # [CODE for VALIDATION every epoch]
        net.eval()
        num_iter = len(valid_loader)
        correct = 0 # the number of correct cases
        for i, data in enumerate(tqdm(valid_loader), 0):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            _, true = torch.max(labels.data, 1)
            correct += (predicted == true).sum().item()

            # print statistics
            running_loss += loss.item()

        print('[%d epoch] val loss: %.3f, val acc: %.3f' %
            (epoch + 1, running_loss / num_iter, correct / len(valid_dataset)))

        validation_errors.append(1.0 - correct / len(valid_dataset))
        running_loss = 0.0

    print('Finished Training')
    return net

