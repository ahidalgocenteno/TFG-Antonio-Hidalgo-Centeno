import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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
def test_knn_siamese_network(model, device, train_loader_singles, test_loader_singles):
  # get the accuracy of the siamese with kNN classifier
  model.eval()
  with torch.no_grad():
    # get the embeddings for the test set
    test_embeddings = [] 
    test_embeddings = []
    test_labels = []
    for data, target in test_loader_singles:
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      test_embeddings.append(output)
      test_labels.append(target)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)
    # to cpu
    test_embeddings = test_embeddings.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    # get the embeddings for the train set
    train_embeddings = []
    train_labels = []
    for data, target in train_loader_singles:
      data, target = data.to(device), target.to(device)
      output = model.forward_once(data)
      train_embeddings.append(output)
      train_labels.append(target)
    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)
    # to cpu and convert to numpy
    train_embeddings = train_embeddings.cpu().numpy()
    train_labels = train_labels.cpu().numpy()     

    n_neighbors = 1
    # Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)

    # Predict the class labels for the test set using kNN
    predicted_labels = knn.predict(test_embeddings)
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == test_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

  return accuracy

# test only features
def test_kNN_features(train_loader_features, test_loader_features):
  # get the accuracy of the siamese with kNN classifier
  with torch.no_grad():
    
    # get the embeddings for the train set
    train_embeddings = []
    train_labels = []
    for data, target, features in train_loader_features:
      train_embeddings.append(features)
      train_labels.append(target)
    train_embeddings = torch.cat(train_embeddings)
    train_labels = torch.cat(train_labels)
    # convert to numpy
    train_embeddings = train_embeddings.numpy()
    train_labels = train_labels.numpy()
    
    # get the embeddings for the test set
    test_embeddings = []
    test_labels = []
    for data, target, features in test_loader_features:
      test_embeddings.append(features)
      test_labels.append(target)
    test_embeddings = torch.cat(test_embeddings)
    test_labels = torch.cat(test_labels)
    # convert to numpy
    test_embeddings = test_embeddings.numpy()
    test_labels = test_labels.numpy()

    n_neighbors = 1
    # Train kNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)

    # Predict the class labels for the test set using kNN
    predicted_labels = knn.predict(test_embeddings)
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == test_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

  return accuracy