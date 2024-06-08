import os
import argparse
import pandas as pd
import numpy as np
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
import pickle

def main(args):
    train_directory_path = f'./data/{args.method}/clean_noise/imgs/'
    valid_directory_path = './data/valid_set/images/'
    train_image_list = list_files_in_directory(train_directory_path)
    valid_image_list = list_files_in_directory(valid_directory_path)
    with open(f"./result/small_loss_relabeled/small_loss_0.3_0.7_acc_new_labels.json", "r") as file:
        train_labels = json.load(file)
    with open("./data/valid_set/valdation_data_label.json", "r") as file:
        valid_labels = json.load(file)
    train_label_list = list()
    valid_label_list = list()
    
    for image in train_image_list:
        # train_label_list.append(train_labels[image.split(".")[0]])
        train_label_list.append(train_labels[image])
    for image in valid_image_list:
        image_name = "_".join(image.split("_")[1:]).split(".")[0]
        valid_label_list.append(valid_labels[image_name])
        
    # data transformation: PILImge to Torch tensor, and conducting normalization
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_base_path = f"./data/{args.method}/clean_noise/imgs"
    valid_base_path = "./data/valid_set/images/"
    # 데이터셋 인스턴스 생성
    train_dataset = ImageDataset(train_base_path, train_image_list, np.array(train_label_list, dtype=np.float16), transform)
    valid_dataset = ImageDataset(valid_base_path, valid_image_list, np.array(valid_label_list, dtype=np.float16), transform)

    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=4,
    num_workers=2,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=4, 
        num_workers=2,
    )
    
    num_class = 10
    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 10)  # 10개의 클래스가 있다고 가정
    
    # loss function for classification
    criterion = nn.CrossEntropyLoss()
    criterion

    # optimization for mini-batch stochastic gradient descent (SGD)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)

    # gpu device setup
    device = 'cuda'
    net.to(device)
    
    net = train(net, train_dataset, train_loader, valid_dataset, valid_loader, 
                                          device, criterion, optimizer, lr_scheduler, args)
    
    # new_labels, best_metric = relabel(net, train_dataset, train_loader, valid_dataset, valid_loader, device, criterion, optimizer, lr_scheduler, early_stop_epochs, args)
    
    # new_label_dict = dict()
    # for i, l in zip(train_image_list, new_labels):
    #     new_label_dict[i] = l
    # with open(f"./result/{args.method}_{args.early_ratio}_{args.confidence_threshold}_{args.early_stop_criteria}_new_labels.json", "w") as f:
    #     json.dump(new_label_dict, f)
        
    # return best_metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser('VQL Training')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch_size')
    parser.add_argument('--seed', '-s', type=int, default=42, help='random seed')
    parser.add_argument("--total_epochs", '-e',type=int, default=30)
    parser.add_argument("--method", '-m', type=str, default="gmm")
    parser.add_argument("--early_ratio", type=float, default=0.1)
    parser.add_argument("--confidence_threshold", "-c", type=float, default=0.2)
    parser.add_argument("--early_stop_criteria", '-ec', type=str, default="acc")
    parser.add_argument("--early_stop", '-es', type=int, default=5)
    

    args = parser.parse_args()
    
    method_list = ["small_loss"]
    early_list = [0.1]
    confidence_list = [0.2]
    
    method_result_list = list()
    early_ratio_list = list()
    confidence_threshold_list = list()
    criteria_list = list()
    args.early_stop_criteria = "acc"
    for method in method_list:
        for early_ratio in early_list:
            for confidence_threshold in confidence_list:
                args.method = method
                args.early_ratio = early_ratio
                args.confidence_threshold = confidence_threshold
                df = pd.DataFrame()
                method_result_list.append(method)
                early_ratio_list.append(early_ratio)
                confidence_threshold_list.append(confidence_threshold)
                best_metric = main(args)
                # criteria_list.append(best_metric)
                
                # df["method"] = method_result_list
                # df["warmup_ratio"] = early_ratio_list
                # df["confidence_threshold"] = confidence_threshold_list
                # df["early_stop_criteria"] = criteria_list

                # df.to_csv(f"./result/{args.early_stop_criteria}_result.csv", index=False)
    