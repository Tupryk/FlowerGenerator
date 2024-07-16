import torch
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder, Diffuser
from diffuser import sample

IMAGE_DIMS = 64


def generate_flower():
    diffuser = Diffuser()
    diffuser.load_state_dict(torch.load('./models/diffuser.pth'))
    diffuser.eval()
    latent_im = sample(diffuser, n_samples=1)

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('./models/autoencoder.pth'))
    autoencoder.eval()

    image = autoencoder.dencode(latent_im).detach().cpu().numpy()

    image = image.reshape(IMAGE_DIMS, IMAGE_DIMS, 3)
    image = (image + 1.) * .5
    return image


if __name__ == "__main__":
    image = generate_flower()
    plt.imshow(image)
    plt.show()
