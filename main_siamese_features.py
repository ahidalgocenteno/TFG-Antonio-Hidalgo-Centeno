import torch
from torchvision import datasets, transforms, utils

import json
import collections

from utils.helper_utils import imshow

from networks.siamese_net import siamese_convolutional_net, siamese_recurrent_net
from utils.data_utils import SiameseNetworkDatasetFeatures, load_datos_parciales
from utils.train_test_utils import set_device, train_siamese_with_features

from utils.seed import seed_everything


if __name__ == '__main__':
    # Seed
    seed_everything(42, benchmark=False)

    spectrograms_dir = "Data/images_original/"
    folder_names = ['Data/train/', 'Data/test/', 'Data/val/']
    train_dir = folder_names[0]
    test_dir = folder_names[1]
    val_dir = folder_names[2]

    train_dataset = datasets.ImageFolder(train_dir,transforms.Compose([transforms.ToTensor(),]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True, num_workers=0)

    val_dataset = datasets.ImageFolder(val_dir,transforms.Compose([transforms.ToTensor(),]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=25, shuffle=True, num_workers=0)

    # datos parciales
    data_per_class = [80, 50, 10, 5, 1]
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
            train_parcial_loader = torch.utils.data.DataLoader(train_parcial_dataset, batch_size=25, shuffle=True, num_workers=0)
        
        # Save dataset in dict
        datasets_parciales[n_per_class] = train_parcial_dataset
        # Save loader in dict
        loaders_parciales[n_per_class] = train_parcial_loader

        print('Partial data for', n_per_class, 'samples per class:', len(train_parcial_dataset), 'train samples,', len(val_dataset), 'validation samples')


    # parciales en siamesa
    siamese_parcial_datasets = {}
    siamese_parcial_loaders = {}
    ratio = 0.5
    data_per_class = [10, 5, 1]

    # val siamese dataset
    print('Siamese Validation Data:')
    siamese_val_dataset = SiameseNetworkDatasetFeatures(val_dataset,transforms.Compose([transforms.ToTensor(),]),ratio=ratio)
    siamese_val_loader = torch.utils.data.DataLoader(siamese_val_dataset, batch_size=25, shuffle=True, num_workers=0)
    print('\n')

    # recorre los diferentes casos de data por clase
    for n_per_class in data_per_class:
        print(f'Data for {n_per_class} images per class:')
        siamese_parcial_datasets[n_per_class] = SiameseNetworkDatasetFeatures(datasets_parciales[n_per_class],transforms.Compose([transforms.ToTensor(),]),ratio=ratio)
        siamese_parcial_loaders[n_per_class] = torch.utils.data.DataLoader(siamese_parcial_datasets[n_per_class], batch_size=25, shuffle=True, num_workers=0)
        print('\n')

    # Muestra un batch de ejemplo
    vis_dataloader = torch.utils.data.DataLoader(siamese_parcial_datasets[5],shuffle=True,num_workers=0,batch_size=8)
    example_batch = next(iter(vis_dataloader))
    # Si la etiqueta = 1, los géneros son diferentes (máxima distancia) Caso contrario etiqueta = 0 (minima distancia)
    concatenated = torch.cat((example_batch[0], example_batch[1]),0)
    # Muestra el batch
    imshow(utils.make_grid(concatenated))
    print(example_batch[2].numpy().reshape(-1))
    
    # device
    device = set_device()
    results = collections.defaultdict(dict)


    # SCNN with features
    print('Training and Testing SCNN with features')
    for n_class,siamese_parcial_loader in siamese_parcial_loaders.items():
        print(f'Training for {n_class} data per class.')
        net = siamese_convolutional_net().to(device)
        train_loss, validation_loss, total_val_acc = train_siamese_with_features(net, device, siamese_parcial_loader,siamese_val_loader, val_loader, loaders_parciales[1], 100)
        results['SCNN'][n_class] = total_val_acc
    
    # save results
    with open('scnn_feat_results.json', 'w') as f:
        json.dump(results['SCNN'], f)

    # SCRNN with features
    print('Training and Testing SCRNN with features')
    for n_class,siamese_parcial_loader in siamese_parcial_loaders.items():
        print(f'Training for {n_class} data per class.')
        net = siamese_recurrent_net().to(device)
        train_loss, validation_loss, total_val_acc = train_siamese_with_features(net, device, siamese_parcial_loader,siamese_val_loader, val_loader, loaders_parciales[1], 100)
        results['SCRNN'][n_class] = total_val_acc
    
    # save results
    with open('scrnn_feat_results.json', 'w') as f:
        json.dump(results['SCRNN'], f)