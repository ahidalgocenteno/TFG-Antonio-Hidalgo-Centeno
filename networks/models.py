import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class convolutional_net(nn.Module):
  def __init__(self):
    super(convolutional_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.fc1 = nn.Linear(in_features=2048, out_features=10)

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)
    self.batchnorm6 = nn.BatchNorm2d(num_features=256)


    self.dropout = nn.Dropout(p=0.3, inplace=False)


  def forward(self, x):
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

    # Fully connected layer 1.
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    x = self.fc1(x)
    x = F.softmax(x,dim=1)

    return x

class recurrent_convolutional_net(nn.Module):
  def __init__(self):
    super(recurrent_convolutional_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.fc1 = nn.Linear(in_features=100, out_features=10)

    self.batchnorm1 = nn.BatchNorm2d(num_features=8)
    self.batchnorm2 = nn.BatchNorm2d(num_features=16)
    self.batchnorm3 = nn.BatchNorm2d(num_features=32)
    self.batchnorm4 = nn.BatchNorm2d(num_features=64)
    self.batchnorm5 = nn.BatchNorm2d(num_features=128)
    self.batchnorm6 = nn.BatchNorm2d(num_features=256)

    self.dropout = nn.Dropout(p=0.3, inplace=False)

    self.RNN = nn.LSTM(input_size = 512, hidden_size = 100, batch_first = True)


  def forward(self, x):
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
    x = x.permute(0, 3, 2, 1) #!!!!!
    x = x.reshape(x.shape[0], -1, x.shape[2] * x.shape[3])

    # Red recurrente
    x, _ = self.RNN(x)

    # Coge todos los batches, el último elemento de la RNN, y las 100 características de ese elemento
    x = x[:, -1, :]

    # Fully connected layer 1.
    x = self.dropout(x)
    x = self.fc1(x)
    x = F.softmax(x,dim=1)

    return x


def train(model, device, train_loader, validation_loader, epochs):
  criterion =  nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #Nadam, lr= .001
  train_loss, validation_loss = [], []
  train_acc, validation_acc = [], []
  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.
      correct, total = 0, 0

      for data, target in train_loader:
        # getting the training set
        data, target = data.to(device), target.to(device)
        # Get the model output (call the model with the data from this batch)
        output = model(data)
        # Zero the gradients out)
        optimizer.zero_grad()
        # Get the Loss
        loss  = criterion(output, target)
        # Calculate the gradients
        loss.backward()
        # Update the weights (using the training step of the optimizer)
        optimizer.step()

        tepochs.set_postfix(loss=loss.item())
        running_loss += loss  # add the loss for this batch

        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

      # append the loss for this epoch
      train_loss.append(running_loss.detach().cpu().item()/len(train_loader))
      train_acc.append(correct/total)

      # evaluate on validation data
      model.eval()
      running_loss = 0.
      correct, total = 0, 0

      for data, target in validation_loader:
        # getting the validation set
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        tepochs.set_postfix(loss=loss.item())
        running_loss += loss.item()
        # get accuracy
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

      validation_loss.append(running_loss/len(validation_loader))
      validation_acc.append(correct/total)

  return train_loss, train_acc, validation_loss, validation_acc


# SIAMESE NETWORK ---

class siamese_recurrent_net(nn.Module):
  def __init__(self):
    """Intitalize neural net layers"""
    super(siamese_recurrent_net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

    # NORMALIZACIÓN DE CAPAS: Ver p.175 libro DL con keras y pytorch
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


# Función de pérdida Siamesa
class PerdidaDistanciaEuclidiana(nn.Module):
    def __init__(self, margin=2.0):
        super(PerdidaDistanciaEuclidiana, self).__init__()
        self.margin = margin

    def forward(self, salida_1, salida_2, target):

      # Calcula la distancia euclideana entre las dos salidas
      distancia_euclideana = F.pairwise_distance(salida_1, salida_2, keepdim = True)

      # Perdida
      loss_contrastive = torch.mean((1-target) * torch.pow(distancia_euclideana, 2) +
                                    (target) * torch.pow(torch.clamp(self.margin - distancia_euclideana, min=0.0), 2))

      return loss_contrastive

# Entrenamiento Siamesa
def train_siamese_network(model, device, train_loader,val_loader, epochs):
  criterion =  PerdidaDistanciaEuclidiana()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #Nadam, lr= .001

  train_loss, validation_loss = [], []

  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.
      correct, total = 0, 0

      for i, (data_1, data_2, target) in enumerate(train_loader, 0):

        # getting the training set
        data_1, data_2, target = data_1.to(device), data_2.to(device), target.to(device)

        # Get the model output (call the model with the data from this batch)
        salida_1, salida_2 = model(data_1, data_2)


        # Zero the gradients out)
        optimizer.zero_grad()
        # Get the Loss
        loss  = criterion(salida_1, salida_2, target)
        # Calculate the gradients
        loss.backward()
        # Update the weights (using the training step of the optimizer)
        optimizer.step()

        tepochs.set_postfix(loss=loss.item())
        running_loss += loss  # add the loss for this batch

      # append the loss for this epoch
      train_loss.append(running_loss.detach().cpu().item()/len(train_loader))

      # evaluate on validation data
      model.eval()
      running_loss = 0.
      correct, total = 0, 0

      for i, (data_1, data_2, target) in enumerate(val_loader, 0):
        # getting the validation set
        data_1,data_2, target = data_1.to(device),data_2.to(device), target.to(device)
        optimizer.zero_grad()
        output_1,output_2 = model(data_1, data_2)
        loss = criterion(output_1, output_2, target)
        tepochs.set_postfix(loss=loss.item())
        running_loss += loss.item()

      validation_loss.append(running_loss/len(val_loader))

  return train_loss, validation_loss