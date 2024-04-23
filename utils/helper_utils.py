import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def plot_audio_wave(y,sample_rate,title):
  print(f'y shape: {np.shape(y)} \n')
  print(f'Frecuencia de muestreo (KHz):{sample_rate} muestras/s \n')
  print(f'Longitud audio: {np.shape(y)[0]/sample_rate:.2f} s')

  plt.figure(figsize=(9, 3))
  librosa.display.waveshow(y=y, sr=sample_rate);
  plt.title(title, fontsize=10)
  plt.ylabel('Amp')
  plt.xlabel('t (s)')
  plt.show()


def plot_loss(train_loss,fname = 'loss.png'):
  epochs = len(train_loss)
  fig,ax = plt.subplots()
  ax.plot(list(range(epochs)), train_loss, label='Train Loss')
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Loss')
  ax.set_title('Epoch vs Loss')
  ax.legend()

  plt.savefig(fname)
  plt.show()


def plot_acc(train_acc,fname = 'acc.png'):
  epochs = len(train_acc)
  fig,ax = plt.subplots()
  ax.plot(list(range(epochs)), train_acc, label='Training Loss')
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Accuracy')
  ax.set_title('Epoch vs Accuracy')
  ax.legend()

  plt.savefig(fname)
  plt.show()



def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_loss_accuracy(train_loss, train_acc, validation_loss, validation_acc):
  epochs = len(train_loss)
  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
  ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Loss')
  ax1.set_title('Epoch vs Loss')
  ax1.legend()

  ax2.plot(list(range(epochs)), train_acc, label='Training Accuracy')
  ax2.plot(list(range(epochs)), validation_acc, label='Validation Accuracy')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('Accuracy')
  ax2.set_title('Epoch vs Accuracy')
  ax2.legend()
  fig.set_size_inches(15.5, 5.5)
  plt.show()
