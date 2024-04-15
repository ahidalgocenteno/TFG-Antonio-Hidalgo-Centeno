import os
import shutil
import random
import glob
import json
import torch
from torchvision import datasets, transforms, make_grid

from utils.nn_helper_utils import imshow
from utils.data_utils import load_datos_parciales
from utils.data_utils import SiameseNetworkDatasetRatiod
from utils.nn_helper_utils import set_device
from utils.nn_helper_utils import train
from utils.nn_helper_utils import train_siamese_network
from networks.models import convolutional_net
from networks.models import recurrent_convolutional_net
from networks.models import siamese_recurrent_net

# Create folder with training, testing and validation data.
random.seed(123)

spectrograms_dir = "Data/images_original/"
folder_names = ['Data/train/', 'Data/test/', 'Data/val/']
train_dir = folder_names[0]
test_dir = folder_names[1]
val_dir = folder_names[2]


# Para cada directorio
for f in folder_names:
  # Si existe el directorio, eliminalo
  if os.path.exists(f):
    shutil.rmtree(f)
    os.mkdir(f)
  # Si no existe, crea el nuevo directorio
  else:
    os.mkdir(f)

# Recorre cada genero
genres = list(os.listdir(spectrograms_dir))
for g in genres:
  # find all images & split in train, test, and validation
  src_file_paths= []
  for im in glob.glob(os.path.join(spectrograms_dir, f'{g}',"*.png"), recursive=True):
    src_file_paths.append(im)
  random.shuffle(src_file_paths)
  test_files = src_file_paths[0:10]
  val_files = src_file_paths[10:20]
  train_files = src_file_paths[20:]


  #  make destination folders for train and test images
  for f in folder_names:
    if not os.path.exists(os.path.join(f + f"{g}")):
      os.mkdir(os.path.join(f + f"{g}"))

  # copy training and testing images over
  for f in train_files:
    shutil.copy(f, os.path.join(os.path.join(train_dir + f"{g}") + '/',os.path.split(f)[1]))
  for f in test_files:
    shutil.copy(f, os.path.join(os.path.join(test_dir + f"{g}") + '/',os.path.split(f)[1]))
  for f in val_files:
    shutil.copy(f, os.path.join(os.path.join(val_dir + f"{g}") + '/',os.path.split(f)[1]))

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


# datos siamesa

# parciales en siamesa
siamese_parcial_datasets = {}
siamese_parcial_loaders = {}

ratio = 0.5

# recorre los diferentes casos de data por clase
for n_class,parcial_dataset in datasets_parciales.items():
  if n_class != 1 and n_class != 1:
    print(f'Data for {n_class} images per class:')
    siamese_parcial_datasets[n_class] = SiameseNetworkDatasetRatiod(parcial_dataset,transforms.Compose([transforms.ToTensor(),]),maximize_ratio=True)
    siamese_parcial_loaders[n_class] = torch.utils.data.DataLoader(siamese_parcial_datasets[n_class], batch_size=25, shuffle=True, num_workers=0)
    print('\n')
# Muestra un batch de ejemplo
vis_dataloader = torch.utils.data.DataLoader(siamese_parcial_datasets[5],shuffle=True,num_workers=0,batch_size=8)
example_batch = next(iter(vis_dataloader))
# Si la etiqueta = 1, los géneros son diferentes (máxima distancia) Caso contrario etiqueta = 0 (minima distancia)
concatenated = torch.cat((example_batch[0], example_batch[1]),0)
# Muestra el batch
imshow(make_grid(concatenated))
print(example_batch[2].numpy().reshape(-1))


results = {'CNN':{},'CRNN':{},'Siamesa Convolucional':{},'Siamesa Recurrente':{}}
# TRAINING CNN
device = set_device()
for n_class,parcial_loader in loaders_parciales.items():
  print(f'Training for {n_class} data per class.')
  net = convolutional_net().to(device)
  train_loss, train_acc, validation_loss, validation_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
  results['CNN'][n_class] = validation_acc[-1]

out_file = open("cnn_parcial_results.json", "w")
json.dump(results['CNN'], out_file, indent = 1)


# TRAINING CRNN 
for n_class,parcial_loader in loaders_parciales.items():
  print(f'Training for {n_class} data per class.')
  net = recurrent_convolutional_net().to(device)
  train_loss, train_acc, validation_loss, validation_acc = train(net, device, loaders_parciales[n_class],val_loader, 100)
  results['CRNN'][n_class] = validation_acc[-1]

# TRAINING SIAMESE CONVOLUTIONAL
# siamese evaluation
for n_class,siamese_parcial_loaders in siamese_parcial_loaders.items():
  print(f'Training for {n_class} data per class.')
  net = siamese_recurrent_net().to(device)
  train_loss, validation_loss = train_siamese_network(net, device, loaders_parciales[n_class]['train'],loaders_parciales[n_class]['val'], 100)
  results['CRNN'][n_class] = validation_acc[-1]

# Print table header
print(f"{'Network':<10}{'1':<10}{'5':<10}{'10':<10}{'50':<10}{'89':<10}")

# Iterate through the outer dictionary
for network, result in results.items():
    # Print key and corresponding subkey values
    print(f"{network:<10}{result[1]:<10}{result[5]:<10}{result[10]:<10}{result[50]:<10}{result[89]:<10}")