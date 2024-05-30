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

def test(model, device, test_loader):
  model.eval()
  correct, total = 0, 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      _, predicted = torch.max(output, 1)
      total += target.size(0)
      correct += (predicted == target).sum().item()
  return correct/total

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

# test
def test_siamese_network(model, device, test_loader_singles, class_samples_loader):
  # get the accuracy of the siamese with kNN classifier
  model.eval()
  with torch.no_grad():
    # get the embeddings for the validation set
    val_embeddings = []
    val_labels = []
    for data, target in test_loader_singles:
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      val_embeddings.append(output)
      val_labels.append(target)
    val_embeddings = torch.cat(val_embeddings)
    val_labels = torch.cat(val_labels)

    # get the embeddings for the class samples
    class_samples_embeddings = []
    class_samples_labels = []
    for data, target in class_samples_loader:
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      class_samples_embeddings.append(output)
      class_samples_labels.append(target)
    class_samples_embeddings = torch.cat(class_samples_embeddings)
    class_samples_labels = torch.cat(class_samples_labels)

    # get the accuracy
    correct = 0
    total = 0
    for i in range(len(val_embeddings)):
      # Repeat val_embeddings[i] to match the shape of class_samples_embeddings
      val_embedding_repeated = val_embeddings[i].repeat(class_samples_embeddings.shape[0], 1)
      # Compute the distances using pairwise_distance
      distances = F.pairwise_distance(val_embedding_repeated, class_samples_embeddings)
      _, predicted = torch.min(distances, 0)
      if class_samples_labels[predicted] == val_labels[i]:
        correct += 1
      total += 1

  return correct/total

def test_siamese_with_features(model, device, test_loader_singles, class_samples_loader):
  model.eval()
  with torch.no_grad():
    # get the embeddings for the validation set
    test_embeddings = []
    test_labels = []
    for data, target, features in test_loader_singles:
      # get data embeddings from siaemse network
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      # append features to the output
      output = torch.cat((output, features), dim=1)
      # append to the list
      test_embeddings.append(output)
      test_labels.append(target)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)

    # get the embeddings for the class samples
    class_samples_embeddings = []
    class_samples_labels = []
    for data, target, features in class_samples_loader:
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      output = torch.cat((output, features), dim=1)
      class_samples_embeddings.append(output)
      class_samples_labels.append(target)
    class_samples_embeddings = torch.cat(class_samples_embeddings)
    class_samples_labels = torch.cat(class_samples_labels)

    # get the accuracy
    correct = 0
    total = 0
    for i in range(len(test_embeddings)):
      # Repeat val_embeddings[i] to match the shape of class_samples_embeddings
      val_embedding_repeated = test_embeddings[i].repeat(class_samples_embeddings.shape[0], 1)
      # Compute the distances using pairwise_distance
      distances = F.pairwise_distance(val_embedding_repeated, class_samples_embeddings)
      _, predicted = torch.min(distances, 0)
      if class_samples_labels[predicted] == test_labels[i]:
        correct += 1
      total += 1

  return correct/total