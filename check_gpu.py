import torch

def check_gpu():
  # Check if GPU is available
  if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    print("GPU activa: ", device_name)
  else:
    print("WARNING : La GPU no está activa.")

if __name__ == "__main__":
  check_gpu()
