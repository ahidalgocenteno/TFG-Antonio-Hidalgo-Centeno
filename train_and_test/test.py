import torch
import torch.nn.functional as F


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

# test only features
def test_kNN_features(test_loader_features, class_samples_loader):
  # get the accuracy of the siamese with kNN classifier
  with torch.no_grad():
    # get the embeddings for the validation set
    test_embeddings = []
    test_labels = []
    for data, target, features in test_loader_features:
      test_embeddings.append(features)
      test_labels.append(target)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)

    # get the embeddings for the class samples
    class_samples_embeddings = []
    class_samples_labels = []
    for data, target, features in class_samples_loader:
      class_samples_embeddings.append(features)
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