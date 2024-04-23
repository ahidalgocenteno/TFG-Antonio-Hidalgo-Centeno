import pickle
import torch
from torchvision import datasets, transforms, utils

from utils.helper_utils import imshow
from utils.data_utils import load_datos_parciales
from utils.data_utils import SiameseNetworkDatasetRatiod

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

  # save val_loader in picklefile
  with open('Data/loaders_datasets/val_loader.p', 'wb') as f:
    pickle.dump(val_loader, f)

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

  # guarda datasets y loaders parciales en picklefile
  with open('Data/loaders_datasets/datasets_parciales.p', 'wb') as f:
      pickle.dump(datasets_parciales, f)
  with open('Data/loaders_datasets/loaders_parciales.p', 'wb') as f:
      pickle.dump(loaders_parciales, f)

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
  imshow(utils.make_grid(concatenated))
  print(example_batch[2].numpy().reshape(-1))

  # guarda datasets y loaders parciales de siamesa en picklefile
  with open('Data/loaders_datasets/siamese_datasets_parciales.p', 'wb') as f:
      pickle.dump(siamese_parcial_datasets, f)
  with open('Data/loaders_datasets/siamese_loaders_parciales.p', 'wb') as f:
      pickle.dump(siamese_parcial_loaders, f)
