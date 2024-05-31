import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp_net(nn.Module):
    def __init__(self):
        super(mlp_net, self).__init__()
        # input of 58 features and output of 10 classes for classification
        self.fc1 = nn.Linear(58, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(p=0.3, inplace=False)

        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x