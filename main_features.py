import torch
from torchvision import datasets, transforms, utils
import json
import collections
import os
from utils.data_utils import DatasetWithFeatures, load_datos_parciales
from utils.seed import seed_everything
from train_and_test.train import set_device
from train_and_test.test import test_kNN_features
from networks.mlp_net import mlp_net

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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = datasets.ImageFolder(val_dir,transforms.Compose([transforms.ToTensor(),]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    test_dataset = datasets.ImageFolder(test_dir,transforms.Compose([transforms.ToTensor(),]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    train_dataset_features = DatasetWithFeatures(train_dataset,transforms.Compose([transforms.ToTensor(),]))
    train_loader_features = torch.utils.data.DataLoader(train_dataset_features, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset_features = DatasetWithFeatures(test_dataset,transforms.Compose([transforms.ToTensor(),]))
    test_loader_features = torch.utils.data.DataLoader(test_dataset_features, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # datos parciales
    data_per_class =  DATA_PER_CLASS
    datasets_parciales = {}
    loaders_parciales = {}

    print('Complete data:', len(train_dataset), 'train samples,', len(val_dataset), 'validation samples')

    # Loop through different cases of data per class
    for n_per_class in data_per_class:
        if n_per_class == 80:
            train_parcial_dataset = train_dataset
            train_parcial_loader = train_loader
        else:
            # call loader for n per class
            train_parcial_dir = load_datos_parciales(n_per_class, train_dir)         
            # Get datasets from directories with ImageFolder
            train_parcial_dataset = datasets.ImageFolder(train_parcial_dir, transforms.Compose([transforms.ToTensor()]))
            # Get loaders
            train_parcial_loader = torch.utils.data.DataLoader(train_parcial_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        # Save dataset in dict
        datasets_parciales[n_per_class] = train_parcial_dataset
        # Save loader in dict
        loaders_parciales[n_per_class] = train_parcial_loader

        print('Partial data for', n_per_class, 'samples per class:', len(train_parcial_dataset), 'train samples,', len(val_dataset), 'validation samples')
  
    # device
    device = set_device()
    results = collections.defaultdict(dict)

    # test kNN with features
    class_sample_dataset = DatasetWithFeatures(datasets_parciales[1], transforms.Compose([transforms.ToTensor(),]))
    class_sample_loader = torch.utils.data.DataLoader(class_sample_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_accuracy = test_kNN_features(train_loader_features, test_loader_features)
    results['kNN'] = test_accuracy

    # save results
    with open(os.path.join(results_dir, 'features_results.json'), 'w') as f:
        json.dump(results, f, indent=1)