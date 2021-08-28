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

    for _ in range(2):
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
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(5,1)

    def forward(self, train_seek, train_service, train_features, test_seek, test_service, test_features):
        difference = abs(train_features - test_features)
        difference = self.fc2(difference)
        return (1 - self.sig(difference)).flatten()

net = Net().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)#, momentum=0.9)

#train your model

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
    return float(num_correct)/float(num_samples)*100

train_accuracy = []
test_accuracy = []
data_length = len(testing)
test_length = round(data_length / 5)
for i in range(5):
    train_training_set = training[(i+1)*test_length:]
    train_testing_set = testing[(i+1)*test_length:]
    labels_set = labels[(i+1)*test_length:]
    if i == 4:
        train_training_set = training[0:i*test_length]
        train_testing_set = testing[0:i*test_length]
        labels_set = labels[0:i*test_length]
    elif i != 0:
        train_training_set = np.concatenate((training[0:i*test_length],training[(i+1)*test_length:]),axis=0)
        train_testing_set = np.concatenate((testing[0:i*test_length],testing[(i+1)*test_length:]),axis=0)
        labels_set = np.concatenate((labels[0:i*test_length],labels[(i+1)*test_length:]),axis=0)

    trainset = CustomTensorDataset(tensors=(train_training_set, train_testing_set, torch.tensor(labels_set,dtype=torch.float)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    test_training_set = training[i*test_length:(i+1)*test_length]
    test_testing_set = testing[i*test_length:(i+1)*test_length]
    test_labels_set = labels[i*test_length:(i+1)*test_length]

    testset = CustomTensorDataset(tensors=(test_training_set, test_testing_set, torch.tensor(test_labels_set,dtype=torch.float)))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)
    
    for epoch in range(7):  # loop over the dataset multiple times

        running_loss = 0.0

        for x1,x2,y in trainloader:
            train_seek_traj = x1[0].to(device)
            train_service_traj = x1[1].to(device)
            train_feat = x1[2].to(device)
            test_seek_traj = x2[0].to(device)
            test_service_traj = x2[1].to(device)
            test_feat = x2[2].to(device)
            labels_y = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(train_seek_traj, train_service_traj, train_feat, test_seek_traj, test_service_traj, test_feat)
            loss = criterion(outputs, labels_y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss))

    #save model
    model_name = "validation_simple_model_" + str(i+1) + ".pth"
    torch.save(net.state_dict(),model_name)
    
    tr_accuracy = check_accuracy(False, trainloader, net)
    te_accuracy = check_accuracy(True, testloader, net)

    train_accuracy.append(tr_accuracy)
    test_accuracy.append(te_accuracy)

end_time = time.perf_counter()
print(f"Training took {end_time - start_time:0.4f} seconds")

print("This is the cross 5 validation with a doubled dataset and 7 epochs.")

new_name = "train_test_simple_accuracies.pickle"
pickle_data = [train_accuracy, test_accuracy]

with open(new_name, 'wb') as f:
    pickle.dump(pickle_data, f)

x = [1, 2, 3, 4, 5]
# Plots the number of sampled indexes against the estimation
plt.plot(x, train_accuracy, color="blue", linewidth = 2)

plt.plot(x, test_accuracy, color="red", linewidth = 2)


plt.xlabel('K-Fold')
plt.ylabel('Accuracy') 
plt.legend(["Training Set", "Testing Set"], loc = "upper right")
plt.title('Training and Testing Accuracy with K-Fold Cross Validation')

plt.savefig('simple_validation_plot.png')
plt.show()