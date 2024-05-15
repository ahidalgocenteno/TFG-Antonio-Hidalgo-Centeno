import torch
import torch.nn as nn
import torch.nn.functional as F

class siamese_convolutional_with_features_net(nn.Module):
  def __init__(self):
    super(siamese_convolutional_with_features_net, self).__init__()
    self.cnn = siamese_convolutional_net()
    self.fc = nn.Sequential(
        nn.Linear(59, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

  def forward(self, input1, input2, features1, features2):
      # Pass the inputs through the CNN
      spectogram_output1 = self.cnn(input1)
      spectogram_output2 = self.cnn(input2)

      # Pass the features through the fully connected layers
      features_output1 = self.fc(features1)
      features_output2 = self.fc(features2)

      # Concatenate the CNN outputs and the features outputs
      output1 = torch.cat((spectogram_output1, features_output1), dim=1)
      output2 = torch.cat((spectogram_output2, features_output2), dim=1)

      return output1,output2
  
class siamese_recurrent_with_features_net(nn.Module):
  def __init__(self):
    super(siamese_recurrent_with_features_net, self).__init__()
    self.rnn = siamese_recurrent_net()
    self.fc = nn.Sequential(
        nn.Linear(59, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

  def forward(self, input1, input2, features1, features2):
      # Pass the inputs through the RNN
      spectogram_output1 = self.rnn(input1)
      spectogram_output2 = self.rnn(input2)

      # Pass the features through the fully connected layers
      features_output1 = self.fc(features1)
      features_output2 = self.fc(features2)

      # Concatenate the RNN outputs and the features outputs
      output1 = torch.cat((spectogram_output1, features_output1), dim=1)
      output2 = torch.cat((spectogram_output2, features_output2), dim=1)

      return output1,output2

class siamese_convolutional_net(nn.Module):
  def __init__(self):
    super(siamese_convolutional_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.fc1 = nn.Linear(in_features=512, out_features=100)

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)
    self.batchnorm6 = nn.BatchNorm2d(num_features=256)  

    self.dropout = nn.Dropout(p=0.3, inplace=False)

  def forward_once(self, x):
      # Conv layer 1.
      x = self.conv1(x)
      x = self.batchnorm1(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Conv layer 2.
      x = self.conv2(x)
      x = self.batchnorm2(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Conv layer 3.
      x = self.conv3(x)
      x = self.batchnorm3(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Conv layer 4.
      x = self.conv4(x)
      x = self.batchnorm4(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Conv layer 5.
      x = self.conv5(x)
      x = self.batchnorm5(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Conv layer 6.
      x = self.conv6(x)
      x = self.batchnorm6(x)
      x = F.relu(x)
      x = F.max_pool2d(x, kernel_size=2)

      # Flatten
      x = torch.flatten(x, 1)

      # Fully connected layer 1.
      x = self.fc1(x)

      return x
  
  def forward(self, entrada_1, entrada_2):
    # Llama al modelo con cada una de las entradas
    salida_1 = self.forward_once(entrada_1)
    salida_2 = self.forward_once(entrada_2)

    return salida_1,salida_2

class siamese_recurrent_net(nn.Module):
  def __init__(self):
    super(siamese_recurrent_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)
    self.batchnorm6 = nn.BatchNorm2d(num_features=256)

    self.dropout = nn.Dropout(p=0.3, inplace=False)

    self.RNN = nn.LSTM(input_size = 512, hidden_size = 100, batch_first = True)

  def forward_once(self, x):
    # Conv layer 1.
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 2.
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 3.
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 4.
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 5.
    x = self.conv5(x)
    x = self.batchnorm5(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Conv layer 6.
    x = self.conv6(x)
    x = self.batchnorm6(x)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)

    # Redimensionamiento de los datos
    x = x.permute(0, 3, 2, 1)
    x = x.reshape(x.shape[0], -1, x.shape[2] * x.shape[3])

    # Red recurrente
    x, _ = self.RNN(x)

    # Coge todos los batches, el último elemento de la RNN: las 100 características de ese elemento
    x = x[:, -1, :]

    return x

  def forward(self, entrada_1, entrada_2):
    # Llama al modelo con cada una de las entradas
    salida_1 = self.forward_once(entrada_1)
    salida_2 = self.forward_once(entrada_2)

    return salida_1,salida_2