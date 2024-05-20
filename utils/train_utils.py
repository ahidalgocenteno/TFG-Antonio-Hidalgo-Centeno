import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
      print("WARNING : La GPU no está activa.")
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

      # total accuracy
      acc_total = correct/total

  return train_loss, train_acc, validation_loss, validation_acc, acc_total

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

def train_siamese_with_features(model, device, train_loader, val_loader, epochs):
  criterion =  ContrastiveLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #Nadam, lr= .001

  train_loss, validation_loss = [], []

  with tqdm(range(epochs), unit='epoch') as tepochs:
    tepochs.set_description('Training')
    for epoch in tepochs:
      model.train()
      # keep track of the running loss
      running_loss = 0.

      for i, (data_1, data_2, target, features_1, features_2) in enumerate(train_loader, 0):

        # getting the training set
        data_1, data_2, target = data_1.to(device), data_2.to(device), target.to(device)
        features_1, features_2 = features_1.to(device), features_2.to(device)

        # Get the model output (call the model with the data from this batch)
        salida_1, salida_2 = model(data_1, data_2, features_1, features_2)

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

      for i, (data_1, data_2, target, features_1, features_2) in enumerate(val_loader, 0):
        # getting the validation set
        data_1,data_2, target = data_1.to(device),data_2.to(device), target.to(device)
        features_1, features_2 = features_1.to(device), features_2.to(device)
        optimizer.zero_grad()
        output_1,output_2 = model(data_1, data_2, features_1, features_2)
        loss = criterion(output_1, output_2, target)
        tepochs.set_postfix(loss=loss.item())
        running_loss += loss.item()

      validation_loss.append(running_loss/len(val_loader))

  return train_loss, validation_loss


# get the accuracy of the siamese
def get_accuracy_siamese(model, device, val_loader, class_samples_loader):
  model.eval()
  with torch.no_grad():
    # get the embeddings for the validation set
    val_embeddings = []
    val_labels = []
    for data, target in val_loader:
      data, target = data.to(device), target.to(device)
      output = model.get_embedding(data)
      val_embeddings.append(output)
      val_labels.append(target)
    val_embeddings = torch.cat(val_embeddings)
    val_labels = torch.cat(val_labels)

    # get the embeddings for the class samples
    class_samples_embeddings = []
    class_samples_labels = []
    for data, target in class_samples_loader:
      data, target = data.to(device), target.to(device)
      output = model.foward_once(data)
      class_samples_embeddings.append(output)
      class_samples_labels.append(target)
    class_samples_embeddings = torch.cat(class_samples_embeddings)
    class_samples_labels = torch.cat(class_samples_labels)

    # get the accuracy
    correct = 0
    total = 0
    for i in range(len(val_embeddings)):
      distances = torch.norm(val_embeddings[i] - class_samples_embeddings, dim=1)
      _, predicted = torch.min(distances, 0)
      if class_samples_labels[predicted] == val_labels[i]:
        correct += 1
      total += 1

    return correct/total
  

