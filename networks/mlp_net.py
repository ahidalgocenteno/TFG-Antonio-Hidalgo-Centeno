import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp_net(nn.Module):
    def __init__(self):
        super(mlp_net, self).__init__()
        self.fc1 = nn.Linear(58, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
        
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
                
        x = self.fc5(x)

        return x