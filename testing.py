import torch

device =  "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print(torch.version.cuda)
