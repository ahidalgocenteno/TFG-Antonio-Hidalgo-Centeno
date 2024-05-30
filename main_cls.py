import torch
from torchvision import datasets, transforms

import json
import collections

from utils.data_utils import load_datos_parciales
from utils.train_utils import train, set_device

from networks.convolutional_net import convolutional_net
from networks.recurrent_convolutional_net import recurrent_convolutional_net

from utils.helper_utils import plot_loss_accuracy

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

# device
device = set_device()
results = collections.defaultdict(dict)

# CNN 
print('Training and Testing CNN')
for n_class,parcial_loader in loaders_parciales.items():
  print(f'Training for {n_class} data per class.')
  net = convolutional_net().to(device)
  train_loss, train_acc, validation_loss, validation_acc, total_val_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
  results['CNN'][n_class] = total_val_acc
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc, show=False, save=True, fname=f'cnn_{n_class}.png')

with open('cnn_results.json', 'w') as fp:
    json.dump(results['CNN'], fp, indent = 1)

print('Training and Testing  CRNN')
# CRNN
for n_class,parcial_loader in loaders_parciales.items():
  print(f'Training for {n_class} data per class.')
  net = recurrent_convolutional_net().to(device)
  train_loss, train_acc, validation_loss, validation_acc, total_val_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
  results['CRNN'][n_class] = total_val_acc
  plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc, show=False, save=True, fname=f'cnn_{n_class}.png')

with open('crnn_results.json', 'w') as fp:
    json.dump(results['CRNN'], fp, indent = 1)

