import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("WARNING : La GPU no está activa.")
  else:
      device_name = torch.cuda.get_device_name()
      print("GPU activa:", device_name)
  return device


def train_features(model, device, train_loader, validation_loader, epochs):
  criterion =  nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  train_loss, validation_loss = [], []
  train_acc, validation_acc = [], []

  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.
      correct, total = 0, 0

      for _, features, target  in train_loader:
        # getting the training set
        features, target = features.to(device), target.to(device)
        # Get the model output (call the model with the data from this batch)
        output = model(features)
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

      for _, features, target  in validation_loader:
        # getting the validation set
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(features)
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

def train(model, device, train_loader, validation_loader, epochs):
  criterion =  nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
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

# Función de pérdida Siamesa
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, salida_1, salida_2, target):

      # Calcula la distancia euclideana entre las dos salidas
      distancia_euclideana = F.pairwise_distance(salida_1, salida_2, keepdim = True)

      # Perdida
      loss_contrastive = torch.mean((1-target) * torch.pow(distancia_euclideana, 2) +
                                    (target) * torch.pow(torch.clamp(self.margin - distancia_euclideana, min=0.0), 2))

      return loss_contrastive
# Entrenamiento Siamesa
def train_siamese_network(model, device, train_loader_pairs,val_loader_pairs, epochs):
  criterion =  ContrastiveLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) 

  train_loss, validation_loss = [], []

  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.

      for i, (data_1, data_2, target) in enumerate(train_loader_pairs, 0):

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
      train_loss.append(running_loss.detach().cpu().item()/len(train_loader_pairs))

      # evaluate on validation data
      model.eval()
      running_loss = 0.

      for i, (data_1, data_2, target) in enumerate(val_loader_pairs, 0):
        # getting the validation set
        data_1,data_2, target = data_1.to(device),data_2.to(device), target.to(device)
        optimizer.zero_grad()
        output_1,output_2 = model(data_1, data_2)
        loss = criterion(output_1, output_2, target)
        tepochs.set_postfix(loss=loss.item())
        running_loss += loss.item()

      validation_loss.append(running_loss/len(val_loader_pairs))
      
  return train_loss, validation_loss

