import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder, Diffuser
from diffuser import sample, device

IMAGE_DIMS = 128


def generate_flower():
    diffuser = Diffuser().to(device)
    diffuser.load_state_dict(torch.load('./models/diffuser.pth'))
    diffuser.eval()
    latent_im = sample(diffuser, n_samples=1).detach().cpu().numpy()
    print(latent_im)
    latent_im = latent_im.reshape(1, 128, 8, 8)
    latent_im = torch.Tensor(latent_im).to(device)

    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load('./models/autoencoder.pth'))
    autoencoder.eval()

    print(latent_im)
    image = autoencoder.decoder(latent_im)

    image = image.detach().cpu().numpy()
    print(image)
    image = np.maximum(image, 0)[0]
    image = np.transpose(image, (1, 2, 0))
    return image


if __name__ == "__main__":
    image = generate_flower()
    plt.imshow(image)
    plt.show()
