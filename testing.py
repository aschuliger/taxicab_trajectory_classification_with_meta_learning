import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")

start_time = time.perf_counter()

path = 'preprocessed_data'#'test_data'
filenames = glob.glob(path + "/*.pickle")
training_same = []
testing_same = []
labels_same = []
training_diff = []
train_labels_diff = []
for filename in filenames:
    data = pickle.load( open(filename, "rb") )
    training_data = data[0]
    label = data[1]

    indices = random.sample(range(len(training_data)), len(training_data))
    training_data = [training_data[i] for i in indices]
    label = [label[i] for i in indices]

    training_same.append(training_data[0])
    testing_same.append(training_data[1])
    labels_same.append(1)

    for i in range(3):
        training_diff.append(training_data[i+2])
        train_labels_diff.append(label[i+2])

indices = random.sample(range(len(training_diff)), len(training_diff))
training_diff = [training_diff[i] for i in indices]
train_labels_diff = [train_labels_diff[i] for i in indices]

testing_diff = []
labels_diff = []
for i in range((int)(len(training_diff)/2)):
    testing_diff.append(training_diff[i+1])
    training_diff.remove(testing_diff[i])
    if train_labels_diff[i] == train_labels_diff[i+1]:
        labels_diff.append(1)
    else:
        labels_diff.append(0)

training = training_same
testing = testing_same
labels = labels_same

for (train_traj, test_traj, label_traj) in zip(training_diff, testing_diff, labels_diff):
    training.append(train_traj)
    testing.append(test_traj)
    labels.append(label_traj)

indices = random.sample(range(len(training)), len(training))
training = [training[i] for i in indices]
testing = [testing[i] for i in indices]
labels = [labels[i] for i in indices]

class CustomTensorDataset(Dataset):
    
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        train_service_traj = torch.tensor(self.tensors[0][index][1],dtype=torch.float32)
        train_seek_traj = torch.tensor(self.tensors[0][index][0],dtype=torch.float32)
        train_feat = torch.tensor(self.tensors[0][index][2],dtype=torch.float32)
        test_service_traj = torch.tensor(self.tensors[1][index][1],dtype=torch.float32)
        test_seek_traj = torch.tensor(self.tensors[1][index][0],dtype=torch.float32)
        test_feat = torch.tensor(self.tensors[1][index][2],dtype=torch.float32)

        y = self.tensors[2][index]

        return (train_seek_traj, train_service_traj, train_feat), (test_seek_traj, test_service_traj, test_feat), y

    def __len__(self):
        return len(self.tensors[0])


#Create Dataset for pytorch training 

num_sub_traj_seek = 70

#define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn_seek = nn.RNN(4,2,batch_first=True)
        self.rnn_service = nn.RNN(4,2,batch_first=True)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        #self.lp = nn.LpDistance(power=2)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Sequential(
            nn.Linear(2*num_sub_traj_seek+2*num_sub_traj_seek+5,1)
            #nn.Linear(10,1)
        )

    def forward(self, train_seek, train_service, train_features, test_seek, test_service, test_features):
        train_seek_rearranged = train_seek.permute(1,0,2,3)
        train_service_rearranged = train_service.permute(1,0,2,3)
        test_seek_rearranged = test_seek.permute(1,0,2,3)
        test_service_rearranged = test_service.permute(1,0,2,3)

        train_seek_output = [self.rnn_seek(cell)[1][0] for cell in train_seek_rearranged]
        train_service_output = [self.rnn_service(cell)[1][0] for cell in train_service_rearranged]
        test_seek_output = [self.rnn_seek(cell)[1][0] for cell in test_seek_rearranged]
        test_service_output = [self.rnn_service(cell)[1][0] for cell in test_service_rearranged]

        #input_tensor = torch.cat(service_output+[emb_fea],axis=1)
        train_input_tensor = torch.cat(train_service_output+[train_features],axis=1)
        train_input_tensor = torch.cat(train_seek_output+[train_input_tensor],axis=1)

        test_input_tensor = torch.cat(test_service_output+[test_features],axis=1)
        test_input_tensor = torch.cat(test_seek_output+[test_input_tensor],axis=1)

        #train_tensor = self.fc2(train_input_tensor)
        #test_tensor = self.fc2(test_input_tensor)

        #outputs = self.similarity(train_input_tensor, test_input_tensor)
        #return self.sig(self.fc2(outputs))
        #train_input_tensor = self.fc2(train_input_tensor)
        #test_input_tensor = self.fc2(test_input_tensor)
        difference = abs(train_input_tensor - test_input_tensor)
        difference = self.fc2(difference)
        return (1 - self.sig(difference)).flatten()
        #return self.sig(F.pairwise_distance(train_input_tensor, test_input_tensor))

testset = CustomTensorDataset(tensors=(training, testing, torch.tensor(labels,dtype=torch.float)))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)


net = Net().to(device)
#net.load_state_dict(torch.load("model_on_all_data_1e2_lr_83.pth"))
net.load_state_dict(torch.load("validation_model_3.pth"))#test_model.pth"))

def check_accuracy(testing, loader, model):
    if testing:
        print("Checking accuracy on test data")
    else:
        print("Checking accuracy on training data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x1, x2, y in loader:
            train_seek_traj = x1[0].to(device)
            train_service_traj = x1[1].to(device)
            train_feat = x1[2].to(device)
            test_seek_traj = x2[0].to(device)
            test_service_traj = x2[1].to(device)
            test_feat = x2[2].to(device)
            labels = y.to(device)
            scores = model(train_seek_traj, train_service_traj, train_feat, test_seek_traj, test_service_traj, test_feat)
            scores[scores > 0.5] = 1
            scores[scores <= 0.5] = 0
            #scores[scores == 1] = 0
            #scores[scores == 2] = 1
            print(len(scores))
            print(len(labels))
            predictions = scores
            print(predictions.shape)
            print(labels.shape)
            #print(predictions == labels)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

            print(f'Label: {labels}')
            print(f'Prediction: {predictions}')

        print(f'Got {num_correct} / {num_samples} with accuracy \ {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

end_time = time.perf_counter()
print(f"Training took {end_time - start_time:0.4f} seconds")

check_accuracy(True, testloader, net)

print("This model was built with RNNs with an output layer of 2 and a learning rate of 0.01")