import pickle
import sys
import pandas as pd
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import math
from scipy.stats import mode
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

device = torch.device("cuda")
start_time = time.perf_counter()

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
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Sequential(
            nn.Linear(2*num_sub_traj_seek+2*num_sub_traj_seek+5,1)
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

        train_input_tensor = torch.cat(train_service_output+[train_features],axis=1)
        train_input_tensor = torch.cat(train_seek_output+[train_input_tensor],axis=1)

        test_input_tensor = torch.cat(test_service_output+[test_features],axis=1)
        test_input_tensor = torch.cat(test_seek_output+[test_input_tensor],axis=1)
        difference = abs(train_input_tensor - test_input_tensor)
        difference = self.fc2(difference)
        return (1 - self.sig(difference)).flatten()

def processs_data(traj_1, traj_2):
    traj1_train = process_traj(traj_1)
    traj2_train = process_traj(traj_2)
    return [traj1_train, traj2_train]

def process_traj(traj):
    max_longitude = 114.51741
    min_longitude = 113.258
    max_latitude = 22.90
    min_latitude = 22.4704

    for i in range(len(traj)):
        if traj[i][1] > max_latitude:
            np.delete(traj, i)
        elif traj[i][0] > max_longitude:
            np.delete(traj, i)
        elif traj[i][1] < min_latitude:
            np.delete(traj, i)
        elif traj[i][0] < min_longitude:
            np.delete(traj, i)

    num_sub_traj_seek = 70
    sequence_length_seek = 400
    num_sub_traj_service = 70
    sequence_length_service = 200
    num_layers = 1

    longitude_distance = max_longitude - min_longitude
    latitude_distance = max_latitude - min_latitude

    def find_cell(traj):
        x = math.floor((traj[1]-min_latitude) / (latitude_distance / 500.0))
        y = math.floor((traj[0]-min_longitude) / (longitude_distance / 500.0))
        return [x, y]

    def calculate_distance(lat1, lat2, long1, long2):
        return math.sqrt(pow(lat2 - lat1, 2) + pow(long2 - long1, 2))

    def calculate_time(date1, date2):
        hours = int(date2[11:13]) - int(date1[11:13])
        mins = int(date2[14:16]) - int(date1[14:16])
        secs = int(date2[17:19]) - int(date1[17:19])
        return hours*60*60 + mins*60 + secs
        
    def get_hour(date1):
        return int(date1[11:13])

    def calculate_speed(lat1, lat2, long1, long2, date1, date2):
        time = calculate_time(date1, date2)
        if time != 0:
            return calculate_distance(lat1, lat2, long1, long2) / time
        return 0

    flag = traj[0][3]
    sub_traj = []
    last_sub_traj = []
    cell_traj = []
    cell_traj_seek = []
    cell_traj_service = []
    grid_array_spec = []
    first_time = traj[0][4]
    last_time = []
    distance = 0
    past_cell = []

    for point in traj:
        if point[3] != flag:
            if len(sub_traj) > 2:
                if flag == 0:
                    cell_traj_seek.append(sub_traj)
                else:
                    cell_traj_service.append(sub_traj)
                last_sub_traj = sub_traj
                sub_traj = []
            elif last_sub_traj != []:
                if flag == 0:
                    cell_traj_service.remove(last_sub_traj)
                else:
                    cell_traj_seek.remove(last_sub_traj)
                sub_traj = last_sub_traj
            else:
                sub_traj = []
            flag = point[3]
        cell = find_cell(point)
        grid_array_spec.append(cell)
        speed = 0
        if past_cell != []:
            speed = calculate_speed(past_cell[0], cell[0], past_cell[1], cell[1], point[4], last_time)
        features = [cell[0], cell[1], get_hour(point[4]), speed]
        sub_traj.append(features)
        last_time = point[4]
        past_cell = [cell[0], cell[1]]
        cell = []
    if len(sub_traj) > 2:
        if flag == 0:
            cell_traj_seek.append(sub_traj)
        else:
            cell_traj_service.append(sub_traj)
    grid_array = grid_array_spec

    training_grids_seek = cell_traj_seek
    training_grids_service = cell_traj_service

    def find_mode_cell(grid):
        x_mode, counts = mode(np.array(grid))
        return np.array(x_mode)[0]

    def average_distances(seek_traj, service_traj):
        seeking_distances = []
        service_distances = []

        for sub_traj in seek_traj:
            distance = 0
            last_cell = sub_traj[0]
            for cell in sub_traj:
                if cell != last_cell:
                    distance = distance + 1
                    last_cell = cell
            seeking_distances.append(distance)

        for sub_traj in service_traj:
            distance = 0
            last_cell = sub_traj[0]
            for cell in sub_traj:
                if cell != last_cell:
                    distance = distance + 1
                    last_cell = cell
            service_distances.append(distance)
            
        avg_seeking = 0
        avg_service = 0
        if len(seeking_distances) != 0:
            avg_seeking = sum(seeking_distances) / len(seeking_distances)
        if len(service_distances) != 0:
            avg_service = sum(service_distances) / len(service_distances)
        return [avg_seeking, avg_service]

    avg_seeking, avg_service = average_distances(training_grids_seek, training_grids_service)
    longitude_cell, latitude_cell = find_mode_cell(grid_array)
    feature = [longitude_cell, latitude_cell, avg_seeking, avg_service, len(training_grids_service)]
    features = feature

    # This goes through each sub-trajectory and pads it as necessary
    while len(training_grids_seek) > num_sub_traj_seek:
        training_grids_seek.remove(training_grids_seek[len(training_grids_seek)-1])
    while len(training_grids_seek) < num_sub_traj_seek:
        training_grids_seek.append(training_grids_seek[np.random.randint(len(training_grids_seek))])
    for sub_traj in training_grids_seek:
        while len(sub_traj) > sequence_length_seek:
            sub_traj.remove(sub_traj[len(sub_traj)-1])
        while len(sub_traj) < sequence_length_seek:
            sub_traj.append(sub_traj[len(sub_traj)-1])

    while len(training_grids_service) > num_sub_traj_service:
        training_grids_service.remove(training_grids_service[len(training_grids_service)-1])
    while len(training_grids_service) < num_sub_traj_service:
        training_grids_service.append([[-1,-1,0,0]])
    for sub_traj in training_grids_service:
        while len(sub_traj) > sequence_length_service:
            sub_traj.remove(sub_traj[len(sub_traj)-1])
        while len(sub_traj) < sequence_length_service:
            sub_traj.append(sub_traj[len(sub_traj)-1])

    training = [training_grids_seek, training_grids_service, features]

    return training

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
            predictions = scores
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

            print(f'Label: {labels}')
            print(f'Prediction: {predictions}')

        print(f'Got {num_correct} / {num_samples} with accuracy \ {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

def run(data, model):
    trainset = CustomTensorDataset(tensors=([data[0]], [data[1]], torch.tensor([2],dtype=torch.float)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    scores = []
    model.eval()
    with torch.no_grad():
        for x1, x2, y in trainloader:
            train_seek_traj = x1[0].to(device)
            train_service_traj = x1[1].to(device)
            train_feat = x1[2].to(device)
            test_seek_traj = x2[0].to(device)
            test_service_traj = x2[1].to(device)
            test_feat = x2[2].to(device)
            labels = y.to(device)
            scores = model(train_seek_traj, train_service_traj, train_feat, test_seek_traj, test_service_traj, test_feat)
            scores = model(train_seek_traj, train_service_traj, train_feat, test_seek_traj, test_service_traj, test_feat)
            scores[scores > 0.5] = 1
            scores[scores <= 0.5] = 0
    model.train()
    return scores[0].item()


filename = "validate_set.pkl"
training = pickle.load( open(filename, "rb") )
#print(len(training[0]))
filename = "validate_label.pkl"
labels = pickle.load( open(filename, "rb") )

net = Net().to(device)
net.load_state_dict(torch.load("final_model.pth"))#, model_3.pth   #final_model_83.pth"))
train = []
test = []

predictions = []

for pairing in training:
    data = processs_data(pairing[0], pairing[1])
    prediction = run(data, net)
    predictions.append(prediction)
    train.append(data[0])
    test.append(data[1])


print(predictions)

testset = CustomTensorDataset(tensors=(train, test, torch.tensor(labels,dtype=torch.float)))
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

check_accuracy(True, testloader, net)

end_time = time.perf_counter()
difference = (end_time-start_time) / 60
print(f"Training took {difference:0.4f} minutes")
