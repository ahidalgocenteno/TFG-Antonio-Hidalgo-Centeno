import torch
from torchvision import datasets, transforms, utils
import json
import collections
import os
from utils.data_utils import DatasetWithFeatures, load_datos_parciales
from utils.seed import seed_everything
from train_and_test.train import set_device, train_features
from train_and_test.test import test_kNN_features, test_features
from networks.mlp_net import mlp_net
from utils.helper_utils import plot_loss_accuracy

# PARAMETERS
BATCH_SIZE = 25
EPOCHS = 100
DATA_PER_CLASS = [1, 5 , 10, 50, 80]

if __name__ == '__main__':
    # Seed
    seed_everything(42, benchmark=False)

    results_dir = 'results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plots_dir = 'plots/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    spectrograms_dir = "Data/images_original/"
    folder_names = ['Data/train/', 'Data/test/', 'Data/val/']
    train_dir = folder_names[0]
    test_dir = folder_names[1]
    val_dir = folder_names[2]

    train_dataset = datasets.ImageFolder(train_dir,transforms.Compose([transforms.ToTensor(),]))
    test_dataset = datasets.ImageFolder(test_dir,transforms.Compose([transforms.ToTensor(),]))
    val_dataset = datasets.ImageFolder(val_dir,transforms.Compose([transforms.ToTensor(),]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    train_dataset_features = DatasetWithFeatures(train_dataset,transforms.Compose([transforms.ToTensor(),]))

    test_dataset_features = DatasetWithFeatures(test_dataset,transforms.Compose([transforms.ToTensor(),]))
    test_loader_features = torch.utils.data.DataLoader(test_dataset_features, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset_features = DatasetWithFeatures(val_dataset,transforms.Compose([transforms.ToTensor(),]))
    val_loader_features = torch.utils.data.DataLoader(val_dataset_features, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # datos parciales con features
    data_per_class = DATA_PER_CLASS
    train_features_loaders_parciales = {}

    print('Complete data:', len(train_dataset_features), 'train samples,', len(test_dataset_features), 'validation samples')
    for n_class in data_per_class:
        if n_class == 80:
            train_parcial_dataset = train_dataset
            train_features_parcial_loader = train_loader
        else:
            # call loader for n per class
            train_parcial_dir = load_datos_parciales(n_class, train_dir)
            train_parcial_dataset = datasets.ImageFolder(train_parcial_dir,transforms.Compose([transforms.ToTensor(),]))
        train_features_datasets_parcial = DatasetWithFeatures(train_parcial_dataset,transforms.Compose([transforms.ToTensor(),]))
        train_features_loaders_parciales[n_class] = torch.utils.data.DataLoader(train_features_datasets_parcial, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    

    results = collections.defaultdict(dict)

    # kNN
    for n_class,train_features_parcial_loader in train_features_loaders_parciales.items():
        print('Partial data for', n_class, 'samples per class:', len(train_features_parcial_loader.dataset), 'train samples')
        accuracy = test_kNN_features(train_features_parcial_loader, test_loader_features)
        results['kNN'][n_class] = accuracy

    # Save results
    with open(results_dir + 'knn_features_results.json', 'w') as f:
        json.dump(results['kNN'], f, indent=2)

    # MLP
    net = mlp_net()
    device = set_device()
    net.to(device)

    for n_class,train_features_parcial_loader in train_features_loaders_parciales.items():
        print(f'Training for {n_class} data per class.')
        train_loss, train_acc, validation_loss, validation_acc = train_features(net, device, train_features_parcial_loader, val_loader_features, EPOCHS)
        plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc, show=False, save=True, fname=os.path.join(plots_dir, f'mlp_{n_class}.png'))
        accuracy = test_features(net, device, test_loader_features)
        results['MLP'][n_class] = accuracy

    with open(results_dir + 'mlp_results.json', 'w') as f:
        json.dump(results['MLP'], f, indent=2)
    