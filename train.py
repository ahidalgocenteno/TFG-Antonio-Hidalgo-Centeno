import torch
from torchvision import datasets, transforms, utils

import json

from utils.helper_utils import imshow
from utils.data_utils import load_datos_parciales
from utils.data_utils import SiameseNetworkDatasetRatiod
from utils.train_utils import train, train_siamese_network, set_device

from networks.convolutional_net import convolutional_net
from networks.recurrent_convolutional_net import recurrent_convolutional_net
from networks.siamese_net import siamese_recurrent_net

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
  data_per_class = [50, 10, 5, 1]
  datasets_parciales = {}
  loaders_parciales = {}
  datasets_parciales[80] = train_dataset
  loaders_parciales[80] = train_loader

  print('Complete data:', len(train_dataset), 'train samples,', len(val_dataset), 'validation samples')

  # Loop through different cases of data per class
  for n_por_class in data_per_class:
      # call loader for n per class
      train_parcial_dir = load_datos_parciales(n_por_class, train_dir)
      # Get datasets from directories with ImageFolder
      train_parcial_dataset = datasets.ImageFolder(train_parcial_dir, transforms.Compose([transforms.ToTensor()]))
      # Save dataset in dict
      datasets_parciales[n_por_class] = train_parcial_dataset
      # Get loaders and store them in the dictionary
      train_loader = torch.utils.data.DataLoader(train_parcial_dataset, batch_size=25, shuffle=True, num_workers=0)
      # Save loader in dict
      loaders_parciales[n_por_class] = train_loader

      print('Partial data for', n_por_class, 'samples per class:', len(train_parcial_dataset), 'train samples,', len(val_dataset), 'validation samples')

# parciales en siamesa
siamese_parcial_datasets = {}
siamese_parcial_loaders = {}
ratio = 0.5
data_per_class = [80, 50, 10, 5, 1]

# val siamese dataset
print('Siamese Validation Data:')
siamese_val_dataset = SiameseNetworkDatasetRatiod(val_dataset,transforms.Compose([transforms.ToTensor(),]),ratio=ratio)
siamese_val_loader = torch.utils.data.DataLoader(siamese_val_dataset, batch_size=25, shuffle=True, num_workers=0)
print('\n')

# recorre los diferentes casos de data por clase
for n_per_class in data_per_class:
  print(f'Data for {n_per_class} images per class:')
  siamese_parcial_datasets[n_per_class] = SiameseNetworkDatasetRatiod(datasets_parciales[n_per_class],transforms.Compose([transforms.ToTensor(),]),ratio=ratio)
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


device = set_device()
print(device)

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