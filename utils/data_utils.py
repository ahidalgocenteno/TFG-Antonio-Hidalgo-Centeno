import os
import shutil
import glob
import random
import math
from itertools import combinations, product
import torch
import numpy as np
from PIL import Image
import csv

def load_datos_parciales(n_por_clase, path):
    train_folder_name = ['Data/datos_parciales_', str(n_por_clase)]
    train_parcial_dir = ''.join(train_folder_name)

    # If directories exist, remove and create new ones

    if os.path.exists(train_parcial_dir):
        shutil.rmtree(train_parcial_dir)
    os.makedirs(train_parcial_dir)

    genres = os.listdir(path)
    for genre in genres:
        src_file_paths = glob.glob(os.path.join(path, genre, "*.png"))
        src_file_paths.sort() 
        n_files = len(src_file_paths)

        if n_files < n_por_clase:
            raise ValueError(f'ERROR: Not enough data for data per class specified. {n_por_clase} samples per class, but only {n_files} found in the folder.')

        # Select n_por_clase random samples for training
        train_samples = random.sample(src_file_paths, n_por_clase)
        for f in train_samples:
            dest_folder = os.path.join(train_parcial_dir, genre)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(f, os.path.join(dest_folder, os.path.split(f)[1]))

    return train_parcial_dir

def compute_number_of_pairs(N, M):
    # Total images
    total_images = M * N
    # Total number of pairs
    total_pairs = math.comb(total_images,2)
    return total_pairs

class SiameseNetworkDataset(torch.utils.data.Dataset):
  def __init__(self,imageFolderDataset,transform=None):
    self.imageFolderDataset = imageFolderDataset
    self.transform = transform
    self.pairs = self.generate_pairs()

  def generate_pairs(self):
    pairs = []
    # obtiene todas las imagenes y su etiqueta (img[1]) de cada genero en el folder dataset
    # convierte a set para que todos los elementos sean unicos, y obtiene una lista
    genres_labels = sorted(list(set(img[1] for img in self.imageFolderDataset.imgs)))

    # todas las imágenes de cada género
    genre_images_dict = {genre_label: sorted([img for img in self.imageFolderDataset.imgs if img[1] == genre_label]) for genre_label in genres_labels}

    # Generate pairs across all possible combinations of images from different genres
    for genre_pair in combinations(genres_labels, 2):
        # Pairs from different genre
        genre1_images = genre_images_dict[genre_pair[0]]
        genre2_images = genre_images_dict[genre_pair[1]]
        genre_pairs = list(product(genre1_images, genre2_images))
        pairs.extend(genre_pairs)

        same_genre_pairs = list(combinations(genre1_images,2))
        pairs.extend(same_genre_pairs)

    # Sort the pairs after they are generated
    pairs = sorted(pairs)
    return pairs

  def __getitem__(self,index):
    img0_tuple, img1_tuple = self.pairs[index]

    # obtiene cada imagen
    img0 = Image.open(img0_tuple[0])
    img1 = Image.open(img1_tuple[0])

    # combierte a RGB, para quitar el canal alpha
    img0 = img0.convert("RGB")
    img1 = img1.convert("RGB")

    # transforma
    if self.transform is not None:
      img0 = self.transform(img0)
      img1 = self.transform(img1)

    # devuelve 1 si son diferentes, 1 si son iguales
    return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))


  def __len__(self):
    return len(self.pairs)


class SiameseNetworkDatasetRatiod(torch.utils.data.Dataset):
    def __init__(self, imageFolderDataset, transform=None, ratio=0.5, maximize_ratio = False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.ratio = ratio
        self.maximize_ratio = maximize_ratio
        self.pairs = self.generate_pairs()

    def generate_pairs(self):
        pairs = []
        same_genre_pairs = []
        diff_genre_pairs = []
        genres_labels = sorted(list(set(img[1] for img in self.imageFolderDataset.imgs)))
        genre_images_dict = {genre_label: sorted([img for img in self.imageFolderDataset.imgs if img[1] == genre_label]) for genre_label in genres_labels}

        # Calculate the number of samples for each class based on the ratio
        num_samples_per_class = min(len(images) for images in genre_images_dict.values())

        if self.ratio == 1 and num_samples_per_class == 1:
          raise ValueError("Error: Not possible to perform combinations of same class pairs with only one sample per classs.")

        # Generate pairs
        for genre_pair in product(genres_labels,repeat=2):
            genre1_images = genre_images_dict[genre_pair[0]]
            genre2_images = genre_images_dict[genre_pair[1]]
            # separate diff genre and same genre combinations
            if genre_pair[0] == genre_pair[1]:
              same_genre_pairs.extend(list(product(genre1_images, genre2_images)))
            else:
               diff_genre_pairs.extend(list(product(genre1_images, genre2_images)))

        # max ratio to calculate proportion
        max_ratio =  len(same_genre_pairs) / (len(same_genre_pairs) + len(diff_genre_pairs))

        if self.maximize_ratio:
          print('Maximizing numbers of pair in Dataset.')
          self.ratio = max_ratio

        # Calculate the number of pairs to include based on the ratio
        if self.ratio >= max_ratio:
          # the maximum number of pairs is limited by the same genre pairs
          num_same_pairs = len(same_genre_pairs)
          num_different_pairs = int((num_same_pairs - (self.ratio*num_same_pairs))/self.ratio)
        else:
          # the maximum number of pairs is limited by the different genre pairs
          num_different_pairs = len(diff_genre_pairs)
          num_same_pairs = int((num_different_pairs*self.ratio)/(1 - self.ratio))

        print(f"Samples: {num_same_pairs} same pairs, {num_different_pairs} different.")

        pairs.extend(random.sample(same_genre_pairs,num_same_pairs))
        pairs.extend(random.sample(diff_genre_pairs,num_different_pairs))

        return pairs

    def __getitem__(self, index):
        img0_tuple, img1_tuple = self.pairs[index]

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.pairs)
    
class SiameseNetworkDatasetRatiodWithFeatures(torch.utils.data.Dataset):
    def __init__(self, imageFolderDataset, transform=None, ratio=0.5, maximize_ratio = False, features_filename = "Data/features_30_sec.csv"):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.ratio = ratio
        self.maximize_ratio = maximize_ratio
        self.pairs = self.generate_pairs()
        self.features_filename = features_filename
        with open(self.features_filename, mode='r') as f:
          reader = csv.reader(f)
          self.features = {rows[0]:rows[1:] for rows in reader}
  
    def generate_pairs(self):
        pairs = []
        same_genre_pairs = []
        diff_genre_pairs = []
        genres_labels = sorted(list(set(img[1] for img in self.imageFolderDataset.imgs)))
        genre_images_dict = {genre_label: sorted([img for img in self.imageFolderDataset.imgs if img[1] == genre_label]) for genre_label in genres_labels}

        # Calculate the number of samples for each class based on the ratio
        num_samples_per_class = min(len(images) for images in genre_images_dict.values())

        if self.ratio == 1 and num_samples_per_class == 1:
          raise ValueError("Error: Not possible to perform combinations of same class pairs with only one sample per classs.")

        # Generate pairs
        for genre_pair in product(genres_labels,repeat=2):
            genre1_images = genre_images_dict[genre_pair[0]]
            genre2_images = genre_images_dict[genre_pair[1]]
            # separate diff genre and same genre combinations
            if genre_pair[0] == genre_pair[1]:
              same_genre_pairs.extend(list(product(genre1_images, genre2_images)))
            else:
               diff_genre_pairs.extend(list(product(genre1_images, genre2_images)))

        # max ratio to calculate proportion
        max_ratio =  len(same_genre_pairs) / (len(same_genre_pairs) + len(diff_genre_pairs))

        if self.maximize_ratio:
          print('Maximizing numbers of pair in Dataset.')
          self.ratio = max_ratio

        # Calculate the number of pairs to include based on the ratio
        if self.ratio >= max_ratio:
          # the maximum number of pairs is limited by the same genre pairs
          num_same_pairs = len(same_genre_pairs)
          num_different_pairs = int((num_same_pairs - (self.ratio*num_same_pairs))/self.ratio)
        else:
          # the maximum number of pairs is limited by the different genre pairs
          num_different_pairs = len(diff_genre_pairs)
          num_same_pairs = int((num_different_pairs*self.ratio)/(1 - self.ratio))

        print(f"Samples: {num_same_pairs} same pairs, {num_different_pairs} different.")

        pairs.extend(random.sample(same_genre_pairs,num_same_pairs))
        pairs.extend(random.sample(diff_genre_pairs,num_different_pairs))

        return pairs

    def __getitem__(self, index):
        img0_tuple, img1_tuple = self.pairs[index]

        img0_dir = img0_tuple[0]
        img1_dir = img1_tuple[0]

        # get the key in the csv for the file, .wav file
        img0_key = ''.join(os.path.basename(img0_dir).split('.')[0]) + '.wav'
        img1_key = ''.join(os.path.basename(img1_dir).split('.')[0]) + '.wav'
        # convert type metal00099.wav to metal.00099.wav
        img0_key = '.'.join(img0_key.split(img0_key[-9:])) + img0_key[-9:]
        img1_key = '.'.join(img1_key.split(img1_key[-9:])) + img1_key[-9:]
        # get the features for the file to tensor (last element is label, not needed)
        img0_features = torch.from_numpy(np.array(self.features[img0_key][:-1], dtype=np.float32))
        img1_features = torch.from_numpy(np.array(self.features[img1_key][:-1], dtype=np.float32))

        img0 = Image.open(img0_dir)
        img1 = Image.open(img1_dir)

        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)), img0_features, img1_features

    def __len__(self):
        return len(self.pairs)