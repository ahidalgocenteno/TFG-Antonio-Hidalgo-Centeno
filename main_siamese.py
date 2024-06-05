import torch
from torchvision import datasets, transforms, utils
import json
import collections
import os
from networks.siamese_net import siamese_recurrent_net, siamese_convolutional_net
from utils.data_utils import SiameseNetworkDataset, DatasetWithFeatures, load_datos_parciales
from train_and_test.train import train_siamese_network, set_device
from train_and_test.test import test_knn_siamese_network
from utils.helper_utils import plot_loss
from utils.seed import seed_everything

# PARAMETERS
BATCH_SIZE = 25
EPOCHS = 10
DATA_PER_CLASS = [1]
RATIO = 0.25
MAX_RATIO = False

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
    data_per_class = DATA_PER_CLASS
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

    # parciales en siamesa
    siamese_parcial_datasets = {}
    siamese_parcial_loaders = {}
    ratio = 0.25

    # val siamese dataset
    print('Siamese Validation Data:')
    siamese_val_dataset = SiameseNetworkDataset(val_dataset,transforms.Compose([transforms.ToTensor(),]), ratio=RATIO, maximize_ratio=MAX_RATIO)
    siamese_val_loader = torch.utils.data.DataLoader(siamese_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print('\n')
    # test class samples
    test_class_samples_dataset = DatasetWithFeatures(datasets_parciales[1],transforms.Compose([transforms.ToTensor(),]))
    test_class_samples_loader = torch.utils.data.DataLoader(test_class_samples_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    

    # recorre los diferentes casos de data por clase
    for n_per_class in data_per_class:
        print(f'Data for {n_per_class} images per class:')
        siamese_parcial_datasets[n_per_class] = SiameseNetworkDataset(datasets_parciales[n_per_class],transforms.Compose([transforms.ToTensor(),]),ratio=RATIO, maximize_ratio=MAX_RATIO)
        siamese_parcial_loaders[n_per_class] = torch.utils.data.DataLoader(siamese_parcial_datasets[n_per_class], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        print('\n')

    # device
    device = set_device()
    results = collections.defaultdict(dict)


    # SCNN
    print('Training and Testing SCNN')
    for n_class,siamese_parcial_loader in siamese_parcial_loaders.items():
        print(f'Training for {n_class} data per class.')
        net = siamese_convolutional_net().to(device)
        train_loss, validation_loss = train_siamese_network(net, device, siamese_parcial_loader, siamese_val_loader, EPOCHS)
        plot_loss(train_loss, validation_loss, show=False, save=True, fname=os.path.join(plots_dir, f'scnn_loss_{n_class}.png'))
        test_accuracy = test_knn_siamese_network(net, device, train_loader, test_loader)
        results['SCNN'][n_class] = test_accuracy
    
    # save results
    with open(os.path.join(results_dir, 'scnn_results.json'), 'w') as f:
        json.dump(results['SCNN'], f, indent=1)

    # SCRNN
    print('Training and Testing SCRNN')
    for n_class,siamese_parcial_loader in siamese_parcial_loaders.items():
        print(f'Training for {n_class} data per class.')
        net = siamese_recurrent_net().to(device)
        train_loss, validation_loss = train_siamese_network(net, device, siamese_parcial_loader, siamese_val_loader, EPOCHS)
        plot_loss(train_loss, validation_loss, show=False, save=True, fname=os.path.join(plots_dir, f'scrnn_loss_{n_class}.png'))
        test_accuracy = test_knn_siamese_network(net, device, train_loader, test_loader)
        results['SCRNN'][n_class] = test_accuracy

    
    # save results
    with open(os.path.join(results_dir, 'scrnn_results.json'), 'w') as f:
        json.dump(results['SCRNN'], f, indent=1)

    print(results)