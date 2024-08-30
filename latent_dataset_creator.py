import torch
import numpy as np
from models import Autoencoder


data = np.load("./data/arrays.npz")["arr_0"]
data = torch.Tensor(data)

model = Autoencoder()
model.load_state_dict(torch.load('./models/autoencoder.pth'))
model.eval()
latent_data = model.encoder(data)

latent_data = latent_data.detach().cpu().numpy()
print(latent_data.shape)

latent_data = latent_data.reshape(len(latent_data), 128*8*8)
print(latent_data.shape)
np.savez("./data/arrays_latent.npz", latent_data)
