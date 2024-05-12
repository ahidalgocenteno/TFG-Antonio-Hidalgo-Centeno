import pickle
import json
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from networks.models import convolutional_net, recurrent_convolutional_net, siamese_recurrent_net

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("WARNING : La GPU no está activa en este cuaderno.")
  return device

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
def train_siamese_network(model, device, train_loader,val_loader, epochs):
  criterion =  ContrastiveLoss()
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

if __name__=='__main__':
  device = set_device()
  print(device)

  # load datasets
  with open('Data/loaders_datasets/loaders_parciales.p', 'rb') as f:
      loaders_parciales = pickle.load(f)
  with open('Data/loaders_datasets/val_loader.p', 'rb') as f:
      val_loader = pickle.load(f)
  with open('Data/loaders_datasets/siamese_loaders_parciales.p', 'rb') as f:
      siamese_parcial_loaders = pickle.load(f)

  results = {'CNN':{},'CRNN':{},'Siamesa Convolucional':{},'Siamesa Recurrente':{}}

  # CNN evaluation
  for n_class,parcial_loader in loaders_parciales.items():
    print(f'Training for {n_class} data per class.')
    net = convolutional_net().to(device)
    train_loss, train_acc, validation_loss, validation_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
    results['CNN'][n_class] = validation_acc[-1]

  out_file = open("cnn_parcial_results.json", "w")
  json.dump(results['CNN'], out_file, indent = 1)

  # CRNN evaluation
  for n_class,parcial_loader in loaders_parciales.items():
    print(f'Training for {n_class} data per class.')
    net = recurrent_convolutional_net().to(device)
    train_loss, train_acc, validation_loss, validation_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
    results['CRNN'][n_class] = validation_acc[-1]

  # Siamesa Convolucional evaluation
  for n_class,siamese_parcial_loaders in siamese_parcial_loaders.items():
    print(f'Training for {n_class} data per class.')
    net = siamese_recurrent_net().to(device)
    train_loss, validation_loss = train_siamese_network(net, device, loaders_parciales[n_class]['train'],loaders_parciales[n_class]['val'], 100)
    results['CRNN'][n_class] = validation_acc[-1]